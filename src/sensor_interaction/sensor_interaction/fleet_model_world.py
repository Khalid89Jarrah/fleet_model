#!/usr/bin/env python3

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import csv
import os
import threading

# Step level CSV logger, enabled with FLEET_MODEL_STEP_LOG=1
STEP_LOG_PATH = "/opt/fleet_model_ws/models/step_log.csv"
_STEP_LOG_ENABLED = os.environ.get("FLEET_MODEL_STEP_LOG", "0") == "1"

_step_log_lock = threading.Lock()
_step_log_file = None
_step_log_writer = None


def _open_step_log():
    global _step_log_file, _step_log_writer
    write_header = not os.path.exists(STEP_LOG_PATH)
    _step_log_file = open(STEP_LOG_PATH, "a", newline="", buffering=1)
    _step_log_writer = csv.writer(_step_log_file)
    if write_header:
        _step_log_writer.writerow(
            [
                "global_step",
                "env_id",
                "episode",
                "step_in_ep",
                "sim_time",
                "ep_mode",
                "sp_x",
                "sp_y",
                "sp_z",
                "omega_x",
                "omega_y",
                "omega_z",
                "accel_x",
                "accel_y",
                "accel_z",
                "err_x",
                "err_y",
                "err_z",
                "err_norm",
                "action_x",
                "action_y",
                "action_z",
                "r_track_quad",
                "r_track_lin",
                "r_smooth",
                "r_smooth2",
                "r_energy",
                "r_accel",
                "r_band",
                "r_recover",
                "r_omega_pen",
                "r_quiet_tau",
                "r_dc",
                "r_dmg_recover",
                "reward_total",
                "terminated",
                "truncated",
                "term_reason",
            ]
        )
        _step_log_file.flush()


def flush_step_log():
    # Called at end of training. Kept for ppo_train.py compatibility.
    global _step_log_file
    if _step_log_file is not None:
        _step_log_file.flush()
        _step_log_file.close()
        _step_log_file = None


_global_step = 0
_global_step_lock = threading.Lock()


def _inc_global_step():
    global _global_step
    with _global_step_lock:
        _global_step += 1
        return _global_step


_env_id_counter = 0
_env_id_lock = threading.Lock()


class FleetModel(gym.Env):
    # Gym env for drone rate control (FRD body frame).
    _episode_counter = 0
    _ep_counter_lock = threading.Lock()

    # Episode modes and their sampling probabilities
    EPISODE_MODES = [
        "hover",
        "long_hover",
        "slow_ramp",
        "step_jumps",
        "stick_release",
        "sustained_yaw",
        "sustained_rp",
        "coupled",
    ]
    EPISODE_PROBS = [0.22, 0.15, 0.13, 0.10, 0.10, 0.10, 0.10, 0.10]

    # Setpoint magnitudes for each mode (rad/s)
    SLOW_RAMP_MAX = 2.0
    STEP_JUMPS_MAX_RP = 4.0
    STEP_JUMPS_MAX_Y = 4.0
    RELEASE_MAG_RP = (3.0, 4.0)
    RELEASE_MAG_Y = (3.0, 4.0)

    # Sustained maneuver setpoint magnitudes (rad/s) and hold/recover durations (s)
    SUSTAINED_MAG_YAW = (1.5, 3.0)
    SUSTAINED_MAG_RP = (1.5, 3.0)
    SUSTAINED_HOLD_RANGE = (8.0, 15.0)  # seconds at constant sp
    SUSTAINED_RECOVER = (3.0, 6.0)  # seconds back at sp=0 to recover

    # Termination thresholds
    OMEGA_LIMIT_RAD_S = 20.0
    ATTITUDE_LIMIT_DEG = 75.0
    STABILITY_THRESHOLD = 0.05
    MAX_STABLE_DURATION = 2.0  # seconds

    EPISODE_DURATION_S = 20.0

    # I_ERR_MAX matches PX4 deployment clamp (MC_RR_INT_LIM default = 0.30)
    I_ERR_MAX = 0.30

    # Cutoff (Hz) of the angular acceleration low pass, matching IMU_DGYRO_CUTOFF
    DGYRO_CUTOFF_HZ = 20.0

    # |omega| threshold below which the quiet torque penalty is active
    QUIET_OMEGA_THRESHOLD = 0.05

    # Hover thrust range sampled per episode (fast model only)
    HOVER_THRUST_RANGE = (0.50, 0.80)

    # Yaw torque low pass cutoff (Hz), matching the firmware yaw filter
    YAW_TQ_CUTOFF_HZ = 2.0

    # Reward coefficients for the DC error and damage recovery terms
    R_DC_COEF = -1.5  # steady state (DC) error penalty
    R_DC_ALPHA = 0.01  # EMA rate: tau = dt/alpha = 0.4 s at dt=0.004
    R_DMG_COEF = 0.60  # time to null bonus after damage onset
    R_DMG_WINDOW = 500  # ticks the bonus stays active (2 s at dt=0.004)

    def __init__(self, main_node, simulation_step_time=0.004):
        self.main_node = main_node
        self.dt = main_node.DT if hasattr(main_node, "DT") else simulation_step_time

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(12,), dtype=np.float32
        )
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)

        # Precompute control allocation matrix once (depends on YAML, fixed)
        specs = self.main_node.read_motor_specs()
        self.effectiveness_matrix = self.main_node.compute_effectiveness_matrix(specs)
        self.inverse_effectiveness_matrix = self.main_node.compute_psudo_inverse(
            self.effectiveness_matrix
        )

        # Hover thrust in FRD (negative = up), resampled per episode in reset() for the fast model
        self.hover_thrust = -0.78

        # Per instance state
        self.angular_velocity_sp = np.zeros(3, dtype=np.float32)
        self.previous_control_effort = np.zeros(3, dtype=np.float32)
        self.previous_previous_control_effort = np.zeros(3, dtype=np.float32)
        self.previous_angular_velocity = np.zeros(3, dtype=np.float32)
        self._i_err = np.zeros(3, dtype=np.float32)
        self.stable_duration = 0.0
        self._yaw_tq_lp = 0.0  # yaw output filter state
        self._e_dc = np.zeros(3, dtype=np.float32)
        self._steps_since_damage = 1_000_000
        self._e_at_damage = 0.0
        # Low pass state for angular acceleration and previous true omega (reward only)
        self._alpha_lp = np.zeros(3, dtype=np.float32)
        self._prev_omega_true = np.zeros(3, dtype=np.float32)

        hover_ctrl = np.array(
            [0.0, 0.0, 0.0, 0.0, 0.0, self.hover_thrust], dtype=np.float32
        )
        self._hover_pwm = self.main_node.compute_sequential_desaturation(
            hover_ctrl, self.inverse_effectiveness_matrix
        )

        # Per env id (used for step_log tagging and unique RNG seeding)
        global _env_id_counter
        with _env_id_lock:
            self._env_id = _env_id_counter
            _env_id_counter += 1

        # Per env RNG so parallel envs don't share state via global numpy
        self._rng = np.random.default_rng(seed=12345 + self._env_id)

        # Episode state machine for setpoint generator
        self._ep_mode = "hover"
        self._ep_step = 0
        self._ep_segment = []  # list of (start_step, end_step, sp_target_or_callable)
        self._ep_seg_idx = 0

        # Track time since last setpoint step (for the recovery bonus)
        self._steps_since_step = 1_000_000
        self._error_at_step = 0.0
        self._step_in_ep = 0
        self._episode_num = 0

        if _STEP_LOG_ENABLED:
            with _step_log_lock:
                if _step_log_writer is None:
                    _open_step_log()

    # Setpoint generator
    def _pick_episode_mode(self):
        return self._rng.choice(self.EPISODE_MODES, p=self.EPISODE_PROBS)

    def _build_episode_setpoint_plan(self, mode: str):
        n_steps = int(self.EPISODE_DURATION_S / self.dt)
        plan = []

        if mode == "hover":
            plan.append((0, n_steps, np.zeros(3, dtype=np.float32)))

        elif mode == "slow_ramp":
            target1 = self._rng.uniform(
                -self.SLOW_RAMP_MAX, +self.SLOW_RAMP_MAX, 3
            ).astype(np.float32)
            target2 = self._rng.uniform(
                -self.SLOW_RAMP_MAX, +self.SLOW_RAMP_MAX, 3
            ).astype(np.float32)

            def _ramp(target, ramp_s, hold_s, t_now_s):
                if t_now_s < ramp_s:
                    return target * (t_now_s / ramp_s)
                elif t_now_s < ramp_s + hold_s:
                    return target.copy()
                elif t_now_s < 2 * ramp_s + hold_s:
                    t_down = t_now_s - (ramp_s + hold_s)
                    return target * (1.0 - t_down / ramp_s)
                else:
                    return np.zeros(3, dtype=np.float32)

            ramp_s, hold_s = 0.5, 1.0
            ramp_total = int((2 * ramp_s + hold_s) / self.dt)
            half = n_steps // 2

            plan.append((0, ramp_total, lambda t: _ramp(target1, ramp_s, hold_s, t)))
            plan.append((ramp_total, half, np.zeros(3, dtype=np.float32)))
            plan.append(
                (half, half + ramp_total, lambda t: _ramp(target2, ramp_s, hold_s, t))
            )
            plan.append((half + ramp_total, n_steps, np.zeros(3, dtype=np.float32)))

        elif mode == "step_jumps":
            t_step = 0
            while t_step < n_steps:
                seg_dur_s = self._rng.uniform(0.3, 1.0)
                seg_dur = int(seg_dur_s / self.dt)
                if self._rng.random() < 0.3:
                    target = np.zeros(3, dtype=np.float32)
                else:
                    target = self._rng.uniform(
                        np.array(
                            [
                                -self.STEP_JUMPS_MAX_RP,
                                -self.STEP_JUMPS_MAX_RP,
                                -self.STEP_JUMPS_MAX_Y,
                            ]
                        ),
                        np.array(
                            [
                                +self.STEP_JUMPS_MAX_RP,
                                +self.STEP_JUMPS_MAX_RP,
                                +self.STEP_JUMPS_MAX_Y,
                            ]
                        ),
                    ).astype(np.float32)
                plan.append((t_step, min(t_step + seg_dur, n_steps), target))
                t_step += seg_dur

        elif mode == "stick_release":
            t_step = 0
            while t_step < n_steps - 100:
                hover_dur = int(self._rng.uniform(0.2, 0.5) / self.dt)
                plan.append((t_step, t_step + hover_dur, np.zeros(3, dtype=np.float32)))
                t_step += hover_dur
                if t_step >= n_steps:
                    break

                hold_dur = int(self._rng.uniform(0.2, 0.5) / self.dt)
                axis_choice = self._rng.choice(["roll", "pitch", "yaw", "all"])
                target = np.zeros(3, dtype=np.float32)
                if axis_choice in ("roll", "all"):
                    target[0] = self._rng.uniform(
                        *self.RELEASE_MAG_RP
                    ) * self._rng.choice([-1, 1])
                if axis_choice in ("pitch", "all"):
                    target[1] = self._rng.uniform(
                        *self.RELEASE_MAG_RP
                    ) * self._rng.choice([-1, 1])
                if axis_choice in ("yaw", "all"):
                    target[2] = self._rng.uniform(
                        *self.RELEASE_MAG_Y
                    ) * self._rng.choice([-1, 1])
                plan.append((t_step, t_step + hold_dur, target))
                t_step += hold_dur
                if t_step >= n_steps:
                    break

                recover_dur = int(self._rng.uniform(0.5, 1.0) / self.dt)
                plan.append(
                    (t_step, t_step + recover_dur, np.zeros(3, dtype=np.float32))
                )
                t_step += recover_dur

            if t_step < n_steps:
                plan.append((t_step, n_steps, np.zeros(3, dtype=np.float32)))

        elif mode == "long_hover":
            # Zero setpoint for the whole episode, so i_err integrates the disturbance
            plan.append((0, n_steps, np.zeros(3, dtype=np.float32)))

        elif mode == "sustained_yaw":
            # Hold a large yaw rate, then recover to hover, repeated
            t_step = 0
            while t_step < n_steps:
                # Hold phase: sustained yaw at large magnitude
                hold_s = self._rng.uniform(*self.SUSTAINED_HOLD_RANGE)
                hold_dur = int(hold_s / self.dt)
                yaw_mag = self._rng.uniform(*self.SUSTAINED_MAG_YAW)
                yaw_sign = self._rng.choice([-1.0, +1.0])
                target = np.array([0.0, 0.0, yaw_sign * yaw_mag], dtype=np.float32)
                plan.append((t_step, min(t_step + hold_dur, n_steps), target))
                t_step += hold_dur
                if t_step >= n_steps:
                    break

                # Recovery phase: back to zero setpoint
                recover_s = self._rng.uniform(*self.SUSTAINED_RECOVER)
                recover_dur = int(recover_s / self.dt)
                plan.append(
                    (
                        t_step,
                        min(t_step + recover_dur, n_steps),
                        np.zeros(3, dtype=np.float32),
                    )
                )
                t_step += recover_dur

        elif mode == "sustained_rp":
            # Hold a large roll or pitch rate (one axis at a time), then recover, repeated
            t_step = 0
            while t_step < n_steps:
                hold_s = self._rng.uniform(*self.SUSTAINED_HOLD_RANGE)
                hold_dur = int(hold_s / self.dt)
                mag = self._rng.uniform(*self.SUSTAINED_MAG_RP)
                sign = self._rng.choice([-1.0, +1.0])
                axis = self._rng.choice([0, 1])  # roll or pitch
                target = np.zeros(3, dtype=np.float32)
                target[axis] = sign * mag
                plan.append((t_step, min(t_step + hold_dur, n_steps), target))
                t_step += hold_dur
                if t_step >= n_steps:
                    break

                recover_s = self._rng.uniform(*self.SUSTAINED_RECOVER)
                recover_dur = int(recover_s / self.dt)
                plan.append(
                    (
                        t_step,
                        min(t_step + recover_dur, n_steps),
                        np.zeros(3, dtype=np.float32),
                    )
                )
                t_step += recover_dur

        elif mode == "coupled":
            # Hold yaw plus one random tilt axis together, then recover, repeated
            t_step = 0
            while t_step < n_steps:
                hold_s = self._rng.uniform(*self.SUSTAINED_HOLD_RANGE)
                hold_dur = int(hold_s / self.dt)
                yaw_mag = self._rng.uniform(*self.SUSTAINED_MAG_YAW)
                tilt_mag = self._rng.uniform(*self.SUSTAINED_MAG_RP)
                tilt_axis = int(self._rng.choice([0, 1]))  # roll OR pitch
                target = np.zeros(3, dtype=np.float32)
                target[tilt_axis] = self._rng.choice([-1.0, 1.0]) * tilt_mag
                target[2] = self._rng.choice([-1.0, 1.0]) * yaw_mag
                plan.append((t_step, min(t_step + hold_dur, n_steps), target))
                t_step += hold_dur
                if t_step >= n_steps:
                    break
                recover_s = self._rng.uniform(*self.SUSTAINED_RECOVER)
                recover_dur = int(recover_s / self.dt)
                plan.append(
                    (
                        t_step,
                        min(t_step + recover_dur, n_steps),
                        np.zeros(3, dtype=np.float32),
                    )
                )
                t_step += recover_dur

        else:
            plan.append((0, n_steps, np.zeros(3, dtype=np.float32)))

        return plan

    def _setpoint_for_step(self, step_idx: int) -> np.ndarray:
        new_sp = None
        for s0, s1, target in self._ep_segment:
            if s0 <= step_idx < s1:
                if callable(target):
                    new_sp = np.asarray(
                        target((step_idx - s0) * self.dt), dtype=np.float32
                    )
                else:
                    new_sp = target.astype(np.float32)
                break
        if new_sp is None:
            new_sp = np.zeros(3, dtype=np.float32)

        # Detect a setpoint "step" (sudden jump > 0.3 rad/s on any axis)
        if step_idx > 0:
            jump = np.max(np.abs(new_sp - self.angular_velocity_sp))
            if jump > 0.3:
                self._steps_since_step = 0
        return new_sp

    # Observation
    def _build_observation(self, omega, accel):
        # 12 dim obs: [omega(3), setpoint(3), accel(3), i_err(3)]
        return np.concatenate(
            [omega, self.angular_velocity_sp, accel, self._i_err]
        ).astype(np.float32)

    # Reset
    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)
        eval_setpoint = options.get("eval_setpoint") if options else None

        # New hover operating point every episode.
        self.hover_thrust = -float(self._rng.uniform(*self.HOVER_THRUST_RANGE))
        hover_ctrl = np.array(
            [0.0, 0.0, 0.0, 0.0, 0.0, self.hover_thrust], dtype=np.float32
        )
        self._hover_pwm = self.main_node.compute_sequential_desaturation(
            hover_ctrl, self.inverse_effectiveness_matrix
        )
        self.main_node.reset_simulation(hover_pwm=self._hover_pwm)

        self.previous_control_effort = np.zeros(3, dtype=np.float32)
        self.previous_previous_control_effort = np.zeros(3, dtype=np.float32)
        self.previous_angular_velocity = np.zeros(3, dtype=np.float32)
        self._i_err = np.zeros(3, dtype=np.float32)
        self.stable_duration = 0.0
        self._alpha_lp = np.zeros(3, dtype=np.float32)
        self._prev_omega_true = np.zeros(3, dtype=np.float32)
        self._steps_since_step = 1_000_000
        self._error_at_step = 0.0
        self._step_in_ep = 0
        self._episode_num += 1
        self._yaw_tq_lp = 0.0
        self._e_dc = np.zeros(3, dtype=np.float32)
        self._steps_since_damage = 1_000_000
        self._e_at_damage = 0.0

        # Read first IMU
        imu_msg = self.main_node.read_imu()
        omega = np.array(
            [
                imu_msg.angular_velocity.x,
                imu_msg.angular_velocity.y,
                imu_msg.angular_velocity.z,
            ],
            dtype=np.float32,
        )
        self.previous_angular_velocity = omega.copy()

        # Build the episode plan
        if eval_setpoint is not None:
            self._ep_mode = "eval"
            n_steps = int(self.EPISODE_DURATION_S / self.dt)
            self._ep_segment = [(0, n_steps, np.array(eval_setpoint, dtype=np.float32))]
        else:
            self._ep_mode = self._pick_episode_mode()
            self._ep_segment = self._build_episode_setpoint_plan(self._ep_mode)

        # Initial setpoint at step 0
        self.angular_velocity_sp = self._setpoint_for_step(0)

        obs = self._build_observation(omega, np.zeros(3, dtype=np.float32))
        info = {
            "episode_mode": self._ep_mode,
            "rate_sp": self.angular_velocity_sp.copy(),
        }
        return obs, info

    # Step
    def step(self, action):
        try:
            action = np.asarray(action, dtype=np.float32)
            action = np.clip(action, -1.0, 1.0)

            # Yaw torque low pass, same as the firmware
            _k_yaw = self.dt / (self.dt + 1.0 / (2.0 * np.pi * self.YAW_TQ_CUTOFF_HZ))
            self._yaw_tq_lp = self._yaw_tq_lp + _k_yaw * (
                float(action[2]) - self._yaw_tq_lp
            )

            # Mix action + hover thrust to motor PWMs (FRD; thrust z negative)
            ctrl = np.array(
                [action[0], action[1], self._yaw_tq_lp, 0.0, 0.0, self.hover_thrust],
                dtype=np.float32,
            )
            pwm = self.main_node.compute_sequential_desaturation(
                ctrl, self.inverse_effectiveness_matrix
            )
            self.main_node.publish_motor_commands(pwm)
            self.main_node.perform_simulation_step(1)

            # Read IMU after step
            imu_msg = self.main_node.read_imu()
            omega = np.array(
                [
                    imu_msg.angular_velocity.x,
                    imu_msg.angular_velocity.y,
                    imu_msg.angular_velocity.z,
                ],
                dtype=np.float32,
            )
            # Angular acceleration: finite difference low passed at DGYRO_CUTOFF_HZ
            raw_alpha = (omega - self.previous_angular_velocity) / self.dt
            _k_alpha = self.dt / (self.dt + 1.0 / (2.0 * np.pi * self.DGYRO_CUTOFF_HZ))
            self._alpha_lp = self._alpha_lp + _k_alpha * (raw_alpha - self._alpha_lp)
            angular_accel = self._alpha_lp.copy()

            # Update setpoint for THIS step (may produce a step jump)
            self._step_in_ep += 1
            sp_before = self.angular_velocity_sp.copy()
            self.angular_velocity_sp = self._setpoint_for_step(self._step_in_ep)

            error = self.angular_velocity_sp - omega
            error_norm = float(np.linalg.norm(error))

            # Update the integral state (clamped) before building the observation
            self._i_err += error * self.dt
            np.clip(self._i_err, -self.I_ERR_MAX, +self.I_ERR_MAX, out=self._i_err)

            # If a setpoint step just happened this tick, record the error then
            if self._steps_since_step == 0:
                self._error_at_step = error_norm

            # Stability tracking (for early success termination)
            if error_norm <= self.STABILITY_THRESHOLD:
                self.stable_duration += self.dt
            else:
                self.stable_duration = 0.0

            # Reward (from the true state on the fast model, so vibration is not penalized)
            omega_true = np.asarray(self.main_node._omega, dtype=np.float32)
            alpha_true = (omega_true - self._prev_omega_true) / self.dt
            self._prev_omega_true = omega_true.copy()
            error_true = self.angular_velocity_sp - omega_true

            reward, r_components = self._compute_reward(
                error=error_true,
                accel=alpha_true,
                action=action,
                prev_action=self.previous_control_effort,
                prev_prev_action=self.previous_previous_control_effort,
                omega=omega_true,
            )

            # Update action history for 2nd derivative computation next tick
            self.previous_previous_control_effort = self.previous_control_effort.copy()
            self.previous_control_effort = action.copy()
            self.previous_angular_velocity = omega.copy()
            self._steps_since_step += 1

            # Count ticks since damage became active
            if getattr(self.main_node, "damage_just_applied", False):
                self._steps_since_damage = 0
            else:
                self._steps_since_damage += 1

            # Termination
            orientation = np.array(
                [
                    imu_msg.orientation.x,
                    imu_msg.orientation.y,
                    imu_msg.orientation.z,
                    imu_msg.orientation.w,
                ],
                dtype=np.float32,
            )
            terminated, term_reason = self._check_termination(
                omega=omega,
                error=error,
                stable_duration=self.stable_duration,
                orientation=orientation,
            )

            sim_time = self.main_node.get_sim_time()
            truncated = (
                False if sim_time is None else sim_time > self.EPISODE_DURATION_S
            )

            # Optional step log
            if _STEP_LOG_ENABLED and _step_log_writer is not None:
                gstep = _inc_global_step()
                row = [
                    gstep,
                    self._env_id,
                    self._episode_num,
                    self._step_in_ep,
                    round(float(sim_time) if sim_time is not None else 0.0, 6),
                    self._ep_mode,
                    *[round(float(v), 6) for v in self.angular_velocity_sp],
                    *[round(float(v), 6) for v in omega],
                    *[round(float(v), 4) for v in angular_accel],
                    *[round(float(v), 6) for v in error],
                    round(error_norm, 6),
                    *[round(float(v), 6) for v in action],
                    round(float(r_components["r_track_quad"]), 6),
                    round(float(r_components["r_track_lin"]), 6),
                    round(float(r_components["r_smooth"]), 6),
                    round(float(r_components["r_smooth2"]), 6),
                    round(float(r_components["r_energy"]), 6),
                    round(float(r_components["r_accel"]), 6),
                    round(float(r_components["r_band"]), 6),
                    round(float(r_components["r_recover"]), 6),
                    round(float(r_components["r_omega_pen"]), 6),
                    round(float(r_components["r_quiet_tau"]), 6),
                    round(float(r_components["r_dc"]), 6),
                    round(float(r_components["r_dmg_recover"]), 6),
                    round(float(reward), 4),
                    int(terminated),
                    int(truncated),
                    term_reason,
                ]
                with _step_log_lock:
                    if _step_log_writer is not None:
                        _step_log_writer.writerow(row)

            obs = self._build_observation(omega, angular_accel)
            return obs, float(reward), bool(terminated), bool(truncated), {}

        except TimeoutError:
            self.main_node.get_logger().warning(
                "Timeout during step, resetting environment."
            )
            obs, info = self.reset()
            return obs, 0.0, False, False, info

    # Reward: sum of tracking, smoothness, energy, in band, recovery, omega, quiet torque, DC and damage terms
    def _compute_reward(
        self, error, accel, action, prev_action, prev_prev_action, omega
    ):
        ex, ey, ez = error[0], error[1], error[2]
        ax, ay, az = accel[0], accel[1], accel[2]
        delta_u = action - prev_action
        delta2_u = action - 2.0 * prev_action + prev_prev_action  # 2nd derivative

        # 1. Per axis tracking error (quadratic and L1)
        r_track_quad = -0.66 * (ex * ex + ey * ey + ez * ez)
        r_track_lin = -0.066 * (abs(ex) + abs(ey) + abs(ez))

        # 2. Smoothness: penalize action changes
        r_smooth = -3.0 * float(delta_u @ delta_u)

        # 3. Second derivative smoothness: penalize action sign flips
        r_smooth2 = -10.0 * float(delta2_u @ delta2_u)

        # 4. Energy
        r_energy = -0.01 * float(action @ action)

        # 5. Angular acceleration penalty
        r_accel = -0.0005 * (ax * ax + ay * ay + az * az)

        # 6. In band bonus for very small tracking error
        eps = 0.02
        r_band = (
            0.10 * np.exp(-((ex / eps) ** 2))
            + 0.10 * np.exp(-((ey / eps) ** 2))
            + 0.10 * np.exp(-((ez / eps) ** 2))
        )

        # 7. Recovery bonus for 50 steps after a setpoint step
        r_recover = 0.0
        if self._steps_since_step < 50 and self._error_at_step > 0.5:
            error_norm = float(np.linalg.norm(error))
            err_reduced = max(0.0, self._error_at_step - error_norm)
            recency = 1.0 - (self._steps_since_step / 50.0)
            r_recover = 0.40 * recency * (err_reduced / max(self._error_at_step, 1e-3))

        # 8. Omega penalty, gated off when the setpoint is large
        sp_norm_sq = float(self.angular_velocity_sp @ self.angular_velocity_sp)
        omega_gate = float(np.exp(-sp_norm_sq / 0.25))  # sigma = 0.5 rad/s
        r_omega_pen = -2.0 * omega_gate * float(omega @ omega)

        # 9. Quiet torque penalty, active only when |omega| is small
        omega_norm_sq = float(omega @ omega)
        if omega_norm_sq < self.QUIET_OMEGA_THRESHOLD**2:
            r_quiet_tau = -2.0 * float(action @ action)
        else:
            r_quiet_tau = 0.0

        # 10. Steady state (DC) error penalty (EMA time constant 0.4 s)
        self._e_dc = (
            1.0 - self.R_DC_ALPHA
        ) * self._e_dc + self.R_DC_ALPHA * np.asarray(error, dtype=np.float32)
        r_dc = self.R_DC_COEF * float(self._e_dc @ self._e_dc)

        # 11. Time to null bonus after damage onset
        err_now = float(np.linalg.norm(error))
        if self._steps_since_damage == 0:
            self._e_at_damage = err_now
        r_dmg_recover = 0.0
        if self._steps_since_damage < self.R_DMG_WINDOW and self._e_at_damage > 0.1:
            recency = 1.0 - (self._steps_since_damage / float(self.R_DMG_WINDOW))
            r_dmg_recover = (
                self.R_DMG_COEF
                * recency
                * max(0.0, self._e_at_damage - err_now)
                / max(self._e_at_damage, 1e-3)
            )

        # 12. Total
        total = (
            r_track_quad
            + r_track_lin
            + r_smooth
            + r_smooth2
            + r_energy
            + r_accel
            + r_band
            + r_recover
            + r_omega_pen
            + r_quiet_tau
            + r_dc
            + r_dmg_recover
        )

        components = {
            "r_track_quad": r_track_quad,
            "r_track_lin": r_track_lin,
            "r_smooth": r_smooth,
            "r_smooth2": r_smooth2,
            "r_energy": r_energy,
            "r_accel": r_accel,
            "r_band": r_band,
            "r_recover": r_recover,
            "r_omega_pen": r_omega_pen,
            "r_quiet_tau": r_quiet_tau,
            "r_dc": r_dc,
            "r_dmg_recover": r_dmg_recover,
        }
        return total, components

    # Termination
    def _check_termination(self, omega, error, stable_duration, orientation):
        if np.linalg.norm(omega) > self.OMEGA_LIMIT_RAD_S:
            return True, "omega_limit"

        if orientation is not None:
            x, y, z, w = orientation
            roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
            pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1.0, 1.0))
            if abs(roll) > np.deg2rad(self.ATTITUDE_LIMIT_DEG) or abs(
                pitch
            ) > np.deg2rad(self.ATTITUDE_LIMIT_DEG):
                return True, "attitude"

        # Early success only in hover episodes; long_hover runs the full duration
        if (
            self._ep_mode == "hover"
            and np.linalg.norm(error) <= self.STABILITY_THRESHOLD
            and stable_duration >= self.MAX_STABLE_DURATION
        ):
            return True, "stable"

        return False, "none"
