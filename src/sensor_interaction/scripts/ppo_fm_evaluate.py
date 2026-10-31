#!/usr/bin/env python3

import argparse
import logging
import os
import sys
import time
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import gymnasium as gym
from datetime import datetime

import sensor_interaction

sys.path.insert(0, "/opt/fleet_model_ws/src/sensor_interaction/scripts")
from ppo_agent import PPOAgent

logging.basicConfig(
    format="[ppo_fm_evaluate] %(levelname)s: %(message)s", level=logging.INFO
)
logger = logging.getLogger("ppo_fm_evaluate")

SAVE_STEP_CSV = False
DT = 0.004

HOVER_TIME = 5.0
STEP_TIME = 8.0
RELEASE_TIME = 2.0
RAMP_TIME = 6.0
SWEEP_TIME = 2.0

STABILITY_THRESH = 0.05
STABLE_REQUIRED = 50

RECOVERY_TARGET = 0.10
RECOVERY_TIMEOUT = 1.0


def quat_to_euler_deg(qx, qy, qz, qw):
    roll = np.degrees(np.arctan2(2 * (qw * qx + qy * qz), 1 - 2 * (qx * qx + qy * qy)))
    pitch = np.degrees(np.arcsin(np.clip(2 * (qw * qy - qz * qx), -1.0, 1.0)))
    yaw = np.degrees(np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz)))
    return roll, pitch, yaw


def profile_hover(n_steps):
    return np.zeros((n_steps, 3), dtype=np.float32)


def profile_step_jumps(n_steps, rng, max_rp=4.0, max_y=4.0):
    # Sequence of constant segments separated by sudden jumps (every 0.3 to 1.0 s).
    sp = np.zeros((n_steps, 3), dtype=np.float32)
    t = 0
    while t < n_steps:
        seg = int(rng.uniform(0.3, 1.0) / DT)
        if rng.random() < 0.3:
            target = np.zeros(3, dtype=np.float32)
        else:
            target = rng.uniform(
                low=[-max_rp, -max_rp, -max_y],
                high=[+max_rp, +max_rp, +max_y],
            ).astype(np.float32)
        end = min(t + seg, n_steps)
        sp[t:end] = target
        t = end
    return sp


def profile_stick_release(n_steps, axis_idx, sign, hold_value=3.5):
    # Hover briefly, hold sp at +/- hold_value on a single axis, then snap to 0.
    # This is the exact failure case from your PX4 crash logs.
    sp = np.zeros((n_steps, 3), dtype=np.float32)
    hover_steps = int(0.3 / DT)
    hold_steps = int(0.4 / DT)
    end_hold = hover_steps + hold_steps
    sp[hover_steps:end_hold, axis_idx] = sign * hold_value
    return sp


def profile_slow_ramp(n_steps, rng, max_amp=2.0, ramp_s=0.5, hold_s=1.0):
    # 0 to target, hold, then back to 0 (smooth).
    sp = np.zeros((n_steps, 3), dtype=np.float32)
    target = rng.uniform(-max_amp, +max_amp, 3).astype(np.float32)
    ramp_steps = int(ramp_s / DT)
    hold_steps = int(hold_s / DT)

    for i in range(n_steps):
        t = i * DT
        if i < ramp_steps:
            sp[i] = target * (t / ramp_s)
        elif i < ramp_steps + hold_steps:
            sp[i] = target
        elif i < 2 * ramp_steps + hold_steps:
            t_down = t - (ramp_s + hold_s)
            sp[i] = target * (1.0 - t_down / ramp_s)
        else:
            sp[i] = 0.0
    return sp


def profile_constant(n_steps, sp_value):
    # Constant setpoint for the sweep grid.
    sp = np.zeros((n_steps, 3), dtype=np.float32)
    sp[:] = np.asarray(sp_value, dtype=np.float32)
    return sp


def run_profile(
    env, agent, sp_profile, label, inv_M, hover_thrust, main_node, save_steps=False
):

    n_steps = sp_profile.shape[0]

    obs, _ = env.reset(options={"eval_setpoint": [0.0, 0.0, 0.0]})

    prev_action = np.zeros(3, dtype=np.float32)
    for _ in range(5):
        env.unwrapped.angular_velocity_sp = np.zeros(3, dtype=np.float32)
        obs, _, _, _, _ = env.step(np.zeros(3, dtype=np.float32))

    step_rows = []
    error_norms = []
    action_buf = {ax: [] for ax in ("roll", "pitch", "yaw")}
    omega_buf = {ax: [] for ax in ("roll", "pitch", "yaw")}
    sp_log = {ax: [] for ax in ("roll", "pitch", "yaw")}
    pwm_means = []
    roll_degs = []
    pitch_degs = []
    raw_actions = []
    sat_count = {ax: 0 for ax in ("roll", "pitch", "yaw")}

    stable_count = 0
    settling_t = None
    max_overshoot = 0.0
    n_crashes = 0

    release_step = None
    recovery_step = None
    error_at_release = None

    for i in range(n_steps):
        env.unwrapped.angular_velocity_sp = sp_profile[i].copy()

        if i > 0:
            jump = float(np.linalg.norm(sp_profile[i] - sp_profile[i - 1]))
            if jump > 0.5:
                if (
                    np.linalg.norm(sp_profile[i]) < 0.1
                    and np.linalg.norm(sp_profile[i - 1]) > 0.5
                ):
                    release_step = i
                    recovery_step = None

        obs = np.asarray(obs, dtype=np.float32).reshape(-1)
        omega = obs[0:3]
        accel = obs[6:9]
        i_err = obs[9:12]
        sp = sp_profile[i]
        error = sp - omega
        e_n = float(np.linalg.norm(error))

        # Obs layout is omega, sp, accel, i_err and sp is overridden by the profile
        obs_for_policy = np.concatenate([omega, sp, accel, i_err]).astype(np.float32)

        action, _ = agent.predict(obs_for_policy, deterministic=True)
        action = np.asarray(action, dtype=np.float32).reshape(3)

        for k, ax_name in enumerate(("roll", "pitch", "yaw")):
            if abs(action[k]) > 0.95:
                sat_count[ax_name] += 1

        ctrl = np.array(
            [action[0], action[1], action[2], 0.0, 0.0, hover_thrust], dtype=np.float32
        )
        pwm = main_node.compute_sequential_desaturation(ctrl, inv_M)

        try:
            imu = main_node.read_imu()
            qx, qy, qz, qw = (
                imu.orientation.x,
                imu.orientation.y,
                imu.orientation.z,
                imu.orientation.w,
            )
        except Exception:
            qx, qy, qz, qw = 0.0, 0.0, 0.0, 1.0
        roll_deg, pitch_deg, _ = quat_to_euler_deg(qx, qy, qz, qw)

        if e_n < STABILITY_THRESH:
            stable_count += 1
            if stable_count >= STABLE_REQUIRED and settling_t is None:
                settling_t = i * DT
        else:
            stable_count = 0

        max_overshoot = max(max_overshoot, e_n)

        if release_step is not None and recovery_step is None:
            steps_since_release = i - release_step
            if steps_since_release == 0:
                error_at_release = e_n
            if e_n < RECOVERY_TARGET and steps_since_release > 0:
                recovery_step = i

        error_norms.append(e_n)
        action_buf["roll"].append(float(action[0]))
        action_buf["pitch"].append(float(action[1]))
        action_buf["yaw"].append(float(action[2]))
        omega_buf["roll"].append(float(omega[0]))
        omega_buf["pitch"].append(float(omega[1]))
        omega_buf["yaw"].append(float(omega[2]))
        sp_log["roll"].append(float(sp[0]))
        sp_log["pitch"].append(float(sp[1]))
        sp_log["yaw"].append(float(sp[2]))
        pwm_means.append(float(np.mean(pwm)))
        roll_degs.append(abs(roll_deg))
        pitch_degs.append(abs(pitch_deg))

        if save_steps:
            step_rows.append(
                {
                    "label": label,
                    "step": i,
                    "sim_time": round(i * DT, 6),
                    "sp_roll": float(sp[0]),
                    "sp_pitch": float(sp[1]),
                    "sp_yaw": float(sp[2]),
                    "omega_x": float(omega[0]),
                    "omega_y": float(omega[1]),
                    "omega_z": float(omega[2]),
                    "error_x": float(error[0]),
                    "error_y": float(error[1]),
                    "error_z": float(error[2]),
                    "error_norm": e_n,
                    "action_roll": float(action[0]),
                    "action_pitch": float(action[1]),
                    "action_yaw": float(action[2]),
                    "pwm_mean": float(np.mean(pwm)),
                }
            )

        prev_action = action.copy()
        obs, _, terminated, truncated, _ = env.step(action)

        if terminated or truncated:
            n_crashes += 1
            logger.info(f"  [{label}] crash #{n_crashes} at step {i}/{n_steps}")
            obs, _ = env.reset(options={"eval_setpoint": [0.0, 0.0, 0.0]})
            stable_count = 0

    tail_start = int(len(error_norms) * 0.75)
    tail = error_norms[tail_start:] or error_norms
    ss_error = float(np.mean(tail))

    tail_omega_r = omega_buf["roll"][tail_start:]
    tail_omega_p = omega_buf["pitch"][tail_start:]
    tail_omega_y = omega_buf["yaw"][tail_start:]

    recovery_time_s = None
    if release_step is not None and recovery_step is not None:
        recovery_time_s = (recovery_step - release_step) * DT
    elif release_step is not None:
        recovery_time_s = -1.0

    summary = {
        "label": label,
        "n_steps": n_steps,
        "sim_time_s": round(n_steps * DT, 3),
        "sp_roll_max": float(np.abs(sp_log["roll"]).max()),
        "sp_pitch_max": float(np.abs(sp_log["pitch"]).max()),
        "sp_yaw_max": float(np.abs(sp_log["yaw"]).max()),
        "settling_time_s": settling_t if settling_t is not None else -1.0,
        "ss_error": ss_error,
        "ss_error_x": float(np.mean(np.abs(tail_omega_r))),
        "ss_error_y": float(np.mean(np.abs(tail_omega_p))),
        "ss_error_z": float(np.mean(np.abs(tail_omega_y))),
        "max_overshoot": max_overshoot,
        "n_crashes": n_crashes,
        "mean_pwm": float(np.mean(pwm_means)),
        "max_roll_deg": float(max(roll_degs)) if roll_degs else 0.0,
        "max_pitch_deg": float(max(pitch_degs)) if pitch_degs else 0.0,
        "sat_roll_pct": 100.0 * sat_count["roll"] / max(n_steps, 1),
        "sat_pitch_pct": 100.0 * sat_count["pitch"] / max(n_steps, 1),
        "sat_yaw_pct": 100.0 * sat_count["yaw"] / max(n_steps, 1),
        "release_time_s": release_step * DT if release_step is not None else -1.0,
        "error_at_release": error_at_release if error_at_release is not None else -1.0,
        "recovery_time_s": recovery_time_s if recovery_time_s is not None else -1.0,
        "mean_action_roll": float(np.mean(action_buf["roll"])),
        "mean_action_pitch": float(np.mean(action_buf["pitch"])),
        "mean_action_yaw": float(np.mean(action_buf["yaw"])),
        "std_action_roll": float(np.std(action_buf["roll"])),
        "std_action_pitch": float(np.std(action_buf["pitch"])),
        "std_action_yaw": float(np.std(action_buf["yaw"])),
    }

    logger.info(
        f"  [{label}] settle={summary['settling_time_s']:.2f}s  "
        f"ss_err={ss_error:.4f}  overshoot={max_overshoot:.3f}  "
        f"crashes={n_crashes}"
    )
    return step_rows, summary, action_buf, omega_buf, sp_log


def plot_overlay(buffers_list, labels, title, ylabel, path, hline=None):
    # Plot multiple time series on the same axes.
    fig, ax = plt.subplots(figsize=(12, 6))
    for buf, lab in zip(buffers_list, labels):
        t = np.arange(len(buf)) * DT
        ax.plot(t, buf, label=lab, lw=0.8)
    if hline is not None:
        ax.axhline(hline, color="k", ls=":", lw=0.6)
        ax.axhline(-hline, color="k", ls=":", lw=0.6)
    ax.axhline(0, color="black", ls="--", lw=0.6, alpha=0.4)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


def plot_hover(action_buf, omega_buf, sp_log, n_steps, path):
    t = np.arange(n_steps) * DT
    fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True)
    fig.suptitle("HOVER TEST , sp = 0 throughout", fontsize=12)

    axes[0].plot(
        t, omega_buf["roll"][:n_steps], color="tab:red", label="omega_x", lw=0.8
    )
    axes[0].plot(
        t, omega_buf["pitch"][:n_steps], color="tab:green", label="omega_y", lw=0.8
    )
    axes[0].plot(
        t, omega_buf["yaw"][:n_steps], color="tab:blue", label="omega_z", lw=0.8
    )
    axes[0].axhline(STABILITY_THRESH, color="gray", ls=":", lw=0.6)
    axes[0].axhline(-STABILITY_THRESH, color="gray", ls=":", lw=0.6)
    axes[0].set_ylabel("omega (rad/s)")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].plot(
        t, action_buf["roll"][:n_steps], color="tab:red", label="a_roll", lw=0.7
    )
    axes[1].plot(
        t, action_buf["pitch"][:n_steps], color="tab:green", label="a_pitch", lw=0.7
    )
    axes[1].plot(
        t, action_buf["yaw"][:n_steps], color="tab:blue", label="a_yaw", lw=0.7
    )
    axes[1].axhline(0, color="k", ls="--", lw=0.5, alpha=0.3)
    axes[1].set_ylabel("action [-1, 1]")
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    axes[2].plot(t, sp_log["roll"][:n_steps], color="tab:red", label="sp_roll")
    axes[2].plot(t, sp_log["pitch"][:n_steps], color="tab:green", label="sp_pitch")
    axes[2].plot(t, sp_log["yaw"][:n_steps], color="tab:blue", label="sp_yaw")
    axes[2].set_ylabel("sp (rad/s)")
    axes[2].set_xlabel("t (s)")
    axes[2].grid(alpha=0.3)
    axes[2].legend()

    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()
    logger.info(f"  hover_plot.png saved")


def plot_release(release_data, path):
    # Overlay all 6 stick release scenarios on one plot.
    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
    fig.suptitle("STICK RELEASE , sp held high -> snap to 0", fontsize=12)

    for label, d in release_data.items():
        n = len(d["error_norms"])
        t = np.arange(n) * DT
        axes[0].plot(t, d["error_norms"], lw=0.9, label=label)

    axes[0].axhline(
        RECOVERY_TARGET, color="k", ls=":", lw=0.6, label=f"recovery={RECOVERY_TARGET}"
    )
    axes[0].set_ylabel("|error|  (rad/s)")
    axes[0].grid(alpha=0.3)
    axes[0].legend(fontsize=8)

    if release_data:
        first = next(iter(release_data.values()))
        n = len(first["sp_roll"])
        t = np.arange(n) * DT
        axes[1].plot(t, first["sp_roll"], color="tab:red", label="sp_roll example")
        axes[1].plot(t, first["sp_pitch"], color="tab:green", label="sp_pitch example")
        axes[1].plot(t, first["sp_yaw"], color="tab:blue", label="sp_yaw example")
        axes[1].set_xlabel("t (s)")
        axes[1].set_ylabel("sp (rad/s)")
        axes[1].grid(alpha=0.3)
        axes[1].legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()
    logger.info(f"  release_plot.png saved")


def plot_step_jumps(step_data, path):
    # Overlay omega tracking for all step_jump episodes.
    fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True)
    fig.suptitle("STEP JUMPS , sudden setpoint changes", fontsize=12)

    for label, d in step_data.items():
        n = len(d["omega_buf"]["roll"])
        t = np.arange(n) * DT
        axes[0].plot(t, d["sp_log"]["roll"], ls="--", lw=0.7, alpha=0.5)
        axes[0].plot(t, d["omega_buf"]["roll"], lw=0.8, label=f"omega_x {label}")
        axes[1].plot(t, d["sp_log"]["pitch"], ls="--", lw=0.7, alpha=0.5)
        axes[1].plot(t, d["omega_buf"]["pitch"], lw=0.8, label=f"omega_y {label}")
        axes[2].plot(t, d["sp_log"]["yaw"], ls="--", lw=0.7, alpha=0.5)
        axes[2].plot(t, d["omega_buf"]["yaw"], lw=0.8, label=f"omega_z {label}")

    axes[0].set_ylabel("roll  (rad/s)")
    axes[0].grid(alpha=0.3)
    axes[0].legend(fontsize=7)
    axes[1].set_ylabel("pitch (rad/s)")
    axes[1].grid(alpha=0.3)
    axes[1].legend(fontsize=7)
    axes[2].set_ylabel("yaw   (rad/s)")
    axes[2].set_xlabel("t (s)")
    axes[2].grid(alpha=0.3)
    axes[2].legend(fontsize=7)

    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()
    logger.info(f"  step_jumps_plot.png saved")


def plot_sweep(df_summary, path):
    df = df_summary[df_summary["label"].str.startswith("sweep_")]
    if df.empty:
        return
    df = df[df["sp_yaw_max"] < 0.01].copy()
    if df.empty:
        return
    df["roll_sp"] = df["label"].str.extract(r"r([+-]?\d+\.\d+)").astype(float)
    df["pitch_sp"] = df["label"].str.extract(r"p([+-]?\d+\.\d+)").astype(float)
    rolls = sorted(df["roll_sp"].unique())
    pitchs = sorted(df["pitch_sp"].unique())
    grid = np.full((len(pitchs), len(rolls)), np.nan)
    for _, r in df.iterrows():
        grid[pitchs.index(r["pitch_sp"]), rolls.index(r["roll_sp"])] = r["ss_error"]

    fig, ax = plt.subplots(figsize=(7, 5))
    vmax = min(np.nanmax(grid), 2.0) if not np.all(np.isnan(grid)) else 2.0
    im = ax.imshow(grid, cmap="RdYlGn_r", aspect="auto", vmin=0, vmax=vmax)
    ax.set_xticks(range(len(rolls)))
    ax.set_xticklabels([f"{v:.2f}" for v in rolls])
    ax.set_yticks(range(len(pitchs)))
    ax.set_yticklabels([f"{v:.2f}" for v in pitchs])
    ax.set_xlabel("Roll setpoint")
    ax.set_ylabel("Pitch setpoint")
    ax.set_title("SWEEP , ss_error heatmap (yaw=0)")
    plt.colorbar(im, ax=ax, label="ss_error (rad/s)")
    for pi in range(len(pitchs)):
        for ri in range(len(rolls)):
            if not np.isnan(grid[pi, ri]):
                ax.text(
                    ri, pi, f"{grid[pi,ri]:.2f}", ha="center", va="center", fontsize=9
                )
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()
    logger.info(f"  sweep_heatmap.png saved")


def write_analysis(
    df, hover_action_buf, hover_omega_buf, out_path, release_data, step_data
):
    L = []
    L.append("=" * 64)
    L.append("  PPO EVALUATION REPORT  (distribution-matched)")
    L.append("=" * 64)

    h = df[df["label"] == "hover"]
    if not h.empty:
        h = h.iloc[0]
        L.append("\n[1] HOVER  (sp = 0)")
        L.append(
            f"  ss_error             = {h['ss_error']:.4f} rad/s   (target < 0.05)"
        )
        L.append(
            f"  ss per axis: roll={h['ss_error_x']:.4f} pitch={h['ss_error_y']:.4f} yaw={h['ss_error_z']:.4f}"
        )
        L.append(f"  max overshoot        = {h['max_overshoot']:.4f}")
        L.append(f"  crashes              = {int(h['n_crashes'])}")
        L.append(f"  mean PWM             = {h['mean_pwm']:.0f}")

        for ax in ("roll", "pitch", "yaw"):
            mean_a = h[f"mean_action_{ax}"]
            verdict = (
                "OK"
                if abs(mean_a) < 0.01
                else ("HIGH" if abs(mean_a) > 0.03 else "low")
            )
            L.append(
                f"  idle action {ax:5s}: mean={mean_a:+.4f}  std={h[f'std_action_{ax}']:.4f}  [{verdict}]"
            )

        L.append(
            f"  HOVER VERDICT: {'PASS' if h['ss_error'] < 0.05 and h['n_crashes'] == 0 else 'FAIL'}"
        )

    sj = df[df["label"].str.startswith("step_jumps_")]
    if not sj.empty:
        L.append("\n[2] STEP_JUMPS  (sp jumps every 0.3-1.0s, +/-4 rad/s)")
        L.append(f"  episodes             = {len(sj)}")
        L.append(f"  total crashes        = {int(sj['n_crashes'].sum())}")
        L.append(f"  mean ss_error        = {sj['ss_error'].mean():.4f}")
        L.append(f"  max overshoot        = {sj['max_overshoot'].max():.3f}")
        L.append(
            f"  mean settling time   = {sj[sj['settling_time_s'] > 0]['settling_time_s'].mean():.3f}s"
        )
        L.append(
            f"  STEP_JUMPS VERDICT: {'PASS' if sj['n_crashes'].sum() == 0 else 'FAIL'}"
        )

    rl = df[df["label"].str.startswith("release_")]
    if not rl.empty:
        L.append(
            "\n[3] STICK_RELEASE  (sp held +/-3.5, snap to 0 , the previous crash mode)"
        )
        L.append(f"  scenarios            = {len(rl)}")
        L.append(f"  total crashes        = {int(rl['n_crashes'].sum())}")
        recovered = rl[rl["recovery_time_s"] > 0]
        failed_recover = rl[rl["recovery_time_s"] == -1]
        L.append(f"  recovered            = {len(recovered)}/{len(rl)}")
        L.append(f"  failed recover       = {len(failed_recover)}/{len(rl)}")
        if not recovered.empty:
            L.append(
                f"  mean recovery time   = {recovered['recovery_time_s'].mean():.3f}s"
            )
            L.append(
                f"  worst recovery time  = {recovered['recovery_time_s'].max():.3f}s"
            )
        for _, r in rl.iterrows():
            ax = r["label"].split("_")[1]
            sg = r["label"].split("_")[2]
            t = r["recovery_time_s"]
            verdict = (
                f"recovered in {t:.2f}s"
                if t > 0
                else "FAILED to recover" if t == -1 else "n/a"
            )
            L.append(
                f"    {ax:5s} {sg:3s}  err@release={r['error_at_release']:.2f}  -> {verdict}"
            )
        L.append(
            f"  STICK_RELEASE VERDICT: {'PASS' if len(failed_recover) == 0 and rl['n_crashes'].sum() == 0 else 'FAIL'}"
        )

    sr = df[df["label"].str.startswith("slow_ramp_")]
    if not sr.empty:
        L.append("\n[4] SLOW_RAMP  (smooth +/-2 rad/s)")
        L.append(f"  episodes             = {len(sr)}")
        L.append(f"  total crashes        = {int(sr['n_crashes'].sum())}")
        L.append(f"  mean ss_error        = {sr['ss_error'].mean():.4f}")
        L.append(f"  max overshoot        = {sr['max_overshoot'].max():.3f}")
        L.append(
            f"  SLOW_RAMP VERDICT: {'PASS' if sr['n_crashes'].sum() == 0 else 'FAIL'}"
        )

    sw = df[df["label"].str.startswith("sweep_")]
    if not sw.empty:
        L.append("\n[5] SWEEP  (3x3x3 grid, constant setpoints)")
        L.append(f"  setpoints            = {len(sw)}")
        L.append(f"  crashes              = {int(sw['n_crashes'].sum())}")
        L.append(f"  mean ss_error        = {sw['ss_error'].mean():.4f}")
        L.append(f"  worst ss_error       = {sw['ss_error'].max():.4f}")
        worst = sw.nlargest(3, "ss_error")[["label", "ss_error", "max_overshoot"]]
        L.append(f"  worst 3 setpoints:")
        for _, r in worst.iterrows():
            L.append(
                f"    {r['label']}: ss={r['ss_error']:.3f} overshoot={r['max_overshoot']:.2f}"
            )
        L.append(
            f"  SWEEP VERDICT: {'PASS' if sw['n_crashes'].sum() == 0 and sw['ss_error'].max() < 0.5 else 'FAIL'}"
        )

    total_crashes = int(df["n_crashes"].sum())
    L.append("\n" + "=" * 64)
    L.append(f"  OVERALL: {total_crashes} crashes across {len(df)} episodes")
    L.append("=" * 64)

    report = "\n".join(L)
    with open(out_path, "w") as f:
        f.write(report)
    for line in L:
        logger.info(line)


def main(args=None):
    t_start = time.monotonic()

    parser = argparse.ArgumentParser(
        description="Evaluate a PPO actor on the fast model"
    )
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="")
    parser.add_argument("--seed", type=int, default=42)
    opts = parser.parse_args(args)
    model_path = opts.model_path
    out_dir = opts.out_dir
    seed = opts.seed

    if not out_dir:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = f"/opt/fleet_model_ws/models/eval_{ts}"
    os.makedirs(out_dir, exist_ok=True)

    rng = np.random.default_rng(seed=int(seed))

    logger.info(f"Output dir : {out_dir}")
    logger.info(f"Model      : {model_path}")
    logger.info(f"RNG seed   : {seed}")
    logger.info(f"Step CSV   : {'ON' if SAVE_STEP_CSV else 'OFF'}")

    agent = PPOAgent.load(model_path, learning_rate=1e-4)
    logger.info("Model loaded")

    env = gym.make("FleetModel-v0")
    inv_M = env.unwrapped.inverse_effectiveness_matrix
    hover_thrust = (
        env.unwrapped.hover_thrust
        if hasattr(env.unwrapped, "hover_thrust")
        else env.unwrapped.Hover_thrust
    )
    main_node_ref = env.unwrapped.main_node

    all_steps = []
    all_summaries = []

    hover_action_buf = None
    hover_omega_buf = None
    hover_sp_log = None
    hover_n = 0

    release_data = {}
    step_data = {}

    try:
        logger.info("\n" + "=" * 64)
        logger.info(f"[1/5] HOVER  ({HOVER_TIME}s)")
        logger.info("=" * 64)
        n_h = int(HOVER_TIME / DT)
        prof = profile_hover(n_h)
        rows, summ, abuf, obuf, splog = run_profile(
            env,
            agent,
            prof,
            "hover",
            inv_M,
            hover_thrust,
            main_node_ref,
            save_steps=SAVE_STEP_CSV,
        )
        if SAVE_STEP_CSV:
            all_steps += rows
        all_summaries.append(summ)
        hover_action_buf, hover_omega_buf, hover_sp_log, hover_n = (
            abuf,
            obuf,
            splog,
            n_h,
        )

        logger.info("\n" + "=" * 64)
        logger.info(f"[2/5] STEP_JUMPS  (3 x {STEP_TIME}s)")
        logger.info("=" * 64)
        for k in range(3):
            n = int(STEP_TIME / DT)
            prof = profile_step_jumps(n, rng=np.random.default_rng(seed + k))
            label = f"step_jumps_{k}"
            rows, summ, abuf, obuf, splog = run_profile(
                env,
                agent,
                prof,
                label,
                inv_M,
                hover_thrust,
                main_node_ref,
                save_steps=SAVE_STEP_CSV,
            )
            if SAVE_STEP_CSV:
                all_steps += rows
            all_summaries.append(summ)
            step_data[label] = {"omega_buf": obuf, "sp_log": splog}

        logger.info("\n" + "=" * 64)
        logger.info(f"[3/5] STICK_RELEASE  (6 x {RELEASE_TIME}s)")
        logger.info("=" * 64)
        for axis_idx, axis_name in [(0, "roll"), (1, "pitch"), (2, "yaw")]:
            for sign, sign_name in [(+1, "pos"), (-1, "neg")]:
                n = int(RELEASE_TIME / DT)
                prof = profile_stick_release(n, axis_idx, sign, hold_value=3.5)
                label = f"release_{axis_name}_{sign_name}"
                rows, summ, abuf, obuf, splog = run_profile(
                    env,
                    agent,
                    prof,
                    label,
                    inv_M,
                    hover_thrust,
                    main_node_ref,
                    save_steps=SAVE_STEP_CSV,
                )
                if SAVE_STEP_CSV:
                    all_steps += rows
                all_summaries.append(summ)

                err_n = []
                for i in range(n):
                    err = prof[i] - np.array(
                        [obuf["roll"][i], obuf["pitch"][i], obuf["yaw"][i]]
                    )
                    err_n.append(float(np.linalg.norm(err)))
                release_data[label] = {
                    "error_norms": err_n,
                    "sp_roll": splog["roll"],
                    "sp_pitch": splog["pitch"],
                    "sp_yaw": splog["yaw"],
                }

        logger.info("\n" + "=" * 64)
        logger.info(f"[4/5] SLOW_RAMP  (3 x {RAMP_TIME}s)")
        logger.info("=" * 64)
        for k in range(3):
            n = int(RAMP_TIME / DT)
            prof = profile_slow_ramp(n, rng=np.random.default_rng(seed + 100 + k))
            label = f"slow_ramp_{k}"
            rows, summ, abuf, obuf, splog = run_profile(
                env,
                agent,
                prof,
                label,
                inv_M,
                hover_thrust,
                main_node_ref,
                save_steps=SAVE_STEP_CSV,
            )
            if SAVE_STEP_CSV:
                all_steps += rows
            all_summaries.append(summ)

        logger.info("\n" + "=" * 64)
        logger.info(f"[5/5] SWEEP  (27 x {SWEEP_TIME}s)")
        logger.info("=" * 64)
        rolls = [-2.0, 0.0, +2.0]
        pitchs = [-2.0, 0.0, +2.0]
        yaws = [-2.0, 0.0, +2.0]
        n_sweep = int(SWEEP_TIME / DT)
        idx = 0
        for r_sp in rolls:
            for p_sp in pitchs:
                for y_sp in yaws:
                    idx += 1
                    sp_val = (r_sp, p_sp, y_sp)
                    label = f"sweep_{idx:02d}_r{r_sp:+.2f}_p{p_sp:+.2f}_y{y_sp:+.2f}"
                    prof = profile_constant(n_sweep, sp_val)
                    rows, summ, _, _, _ = run_profile(
                        env,
                        agent,
                        prof,
                        label,
                        inv_M,
                        hover_thrust,
                        main_node_ref,
                        save_steps=False,
                    )
                    all_summaries.append(summ)

        df = pd.DataFrame(all_summaries)
        df.to_csv(os.path.join(out_dir, "episode_summary.csv"), index=False)
        logger.info(f"\nepisode_summary.csv saved ({len(df)} rows)")

        if SAVE_STEP_CSV and all_steps:
            pd.DataFrame(all_steps).to_csv(
                os.path.join(out_dir, "step_raw.csv"), index=False
            )

        if hover_action_buf is not None:
            plot_hover(
                hover_action_buf,
                hover_omega_buf,
                hover_sp_log,
                hover_n,
                os.path.join(out_dir, "hover_plot.png"),
            )
        if release_data:
            plot_release(release_data, os.path.join(out_dir, "release_plot.png"))
        if step_data:
            plot_step_jumps(step_data, os.path.join(out_dir, "step_jumps_plot.png"))
        plot_sweep(df, os.path.join(out_dir, "sweep_heatmap.png"))

        write_analysis(
            df,
            hover_action_buf or {},
            hover_omega_buf or {},
            os.path.join(out_dir, "analysis.txt"),
            release_data,
            step_data,
        )

        elapsed = (time.monotonic() - t_start) / 60
        logger.info(f"\nEvaluation complete in {elapsed:.1f} minutes")
        logger.info(f"  Results: {out_dir}")

    finally:
        if hasattr(env.unwrapped, "main_node"):
            env.unwrapped.main_node.destroy_node()
        env.close()


if __name__ == "__main__":
    main()
