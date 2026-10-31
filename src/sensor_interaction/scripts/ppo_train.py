#!/usr/bin/env python3

import tensorflow as tf

gpus = tf.config.list_physical_devices("GPU")
if gpus:
    tf.config.experimental.set_memory_growth(gpus[0], True)

import os
import csv
import time
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym

from ppo_agent import PPOAgent, VecEnv
from sensor_interaction.fleet_model_world import FleetModel, flush_step_log

import logging

logging.basicConfig(format="[ppo_train] %(levelname)s: %(message)s", level=logging.INFO)
logger = logging.getLogger("ppo_train")

# Training constants
EVAL_INTERVAL_EPISODES = 1000
START_TRAINING_FROM = 0
TOTAL_EPISODES = 1_000_000
MODEL_BASENAME = "ppo_model"

# PPO hyperparameters
LEARNING_RATE = 1e-4
GAMMA = 0.99
BATCH_SIZE = 512
N_EPOCHS = 10
N_STEPS = 4096
NUM_ENVS = 64

# CSV paths
ROLLOUT_LOG_PATH = "/opt/fleet_model_ws/models/rollout_log.csv"
TRAINING_LOG_PATH = "/opt/fleet_model_ws/models/training_log.csv"

# Rollout log writer (opened once, kept open)
_rollout_log_file = None
_rollout_log_writer = None
_rollout_log_count = 0  # global counter across all rollouts


def _open_rollout_log():
    global _rollout_log_file, _rollout_log_writer
    write_header = not os.path.exists(ROLLOUT_LOG_PATH)
    _rollout_log_file = open(ROLLOUT_LOG_PATH, "a", newline="")
    _rollout_log_writer = csv.writer(_rollout_log_file)
    if write_header:
        _rollout_log_writer.writerow(
            [
                # Timing
                "rollout_idx",
                "total_timesteps",
                # Losses
                "loss_total",
                "loss_pg",
                "loss_vf",
                "loss_ent",
                # Actor state at zero obs
                "log_std_0",
                "log_std_1",
                "log_std_2",
                "mean_zero_0",
                "mean_zero_1",
                "mean_zero_2",
                "std_zero",
                # Buffer statistics: what the agent actually experienced
                "buf_mean_action_0",
                "buf_mean_action_1",
                "buf_mean_action_2",
                "buf_std_action_0",
                "buf_std_action_1",
                "buf_std_action_2",
                "buf_mean_omega_x",
                "buf_mean_omega_y",
                "buf_mean_omega_z",
                "buf_mean_sp_x",
                "buf_mean_sp_y",
                "buf_mean_sp_z",
                "buf_mean_accel_x",
                "buf_mean_accel_y",
                "buf_mean_accel_z",
                "buf_mean_reward",
                "buf_std_reward",
                "buf_min_reward",
                "buf_max_reward",
                # Episode stats from this rollout
                "ep_mean_reward",
                "ep_mean_len",
                "n_episodes",
            ]
        )
        _rollout_log_file.flush()


def _log_rollout(agent, losses, ep_rewards, ep_lengths):
    # Write one row per rollout to rollout_log.csv.
    global _rollout_log_count

    if _rollout_log_writer is None:
        return

    _rollout_log_count += 1

    # Actor state at zero obs
    obs_zero = tf.zeros((1, agent.obs_dim), dtype=tf.float32)  # v9: was hardcoded 9
    mean_zero, std_zero = agent.actor(obs_zero)
    log_std = agent.actor.log_std.numpy()
    mean_zero = mean_zero.numpy()[0]
    std_val = float(std_zero.numpy()[0][0])

    # Buffer statistics, obs layout: [omega(3), sp(3), accel(3), i_err(3)]  (v9, 12 dim)
    buf_obs = agent.buffer.obs  # (n_steps, obs_dim)
    buf_actions = agent.buffer.actions  # (n_steps, 3)
    buf_rewards = agent.buffer.rewards  # (n_steps,), already reward normalized

    _rollout_log_writer.writerow(
        [
            _rollout_log_count,
            agent.num_timesteps,
            # losses
            round(losses.get("train/loss", 0.0), 6),
            round(losses.get("train/policy_loss", 0.0), 6),
            round(losses.get("train/value_loss", 0.0), 6),
            round(losses.get("train/entropy_loss", 0.0), 6),
            # actor at zero obs
            round(float(log_std[0]), 6),
            round(float(log_std[1]), 6),
            round(float(log_std[2]), 6),
            round(float(mean_zero[0]), 6),
            round(float(mean_zero[1]), 6),
            round(float(mean_zero[2]), 6),
            round(std_val, 6),
            # buffer: actions
            round(float(np.mean(buf_actions[:, 0])), 6),
            round(float(np.mean(buf_actions[:, 1])), 6),
            round(float(np.mean(buf_actions[:, 2])), 6),
            round(float(np.std(buf_actions[:, 0])), 6),
            round(float(np.std(buf_actions[:, 1])), 6),
            round(float(np.std(buf_actions[:, 2])), 6),
            # buffer: obs
            round(float(np.mean(buf_obs[:, 0])), 6),  # omega_x
            round(float(np.mean(buf_obs[:, 1])), 6),  # omega_y
            round(float(np.mean(buf_obs[:, 2])), 6),  # omega_z
            round(float(np.mean(buf_obs[:, 3])), 6),  # sp_x
            round(float(np.mean(buf_obs[:, 4])), 6),  # sp_y
            round(float(np.mean(buf_obs[:, 5])), 6),  # sp_z
            round(float(np.mean(buf_obs[:, 6])), 6),  # accel_x
            round(float(np.mean(buf_obs[:, 7])), 6),  # accel_y
            round(float(np.mean(buf_obs[:, 8])), 6),  # accel_z
            # buffer: rewards (normalised)
            round(float(np.mean(buf_rewards)), 6),
            round(float(np.std(buf_rewards)), 6),
            round(float(np.min(buf_rewards)), 6),
            round(float(np.max(buf_rewards)), 6),
            # episode stats
            round(float(np.mean(ep_rewards)) if ep_rewards else 0.0, 4),
            round(float(np.mean(ep_lengths)) if ep_lengths else 0.0, 1),
            len(ep_rewards),
        ]
    )
    _rollout_log_file.flush()


# Evaluation
def evaluate_model(env, agent: PPOAgent, num_episodes: int = 10):
    total_rewards, total_action_mag, total_ang_vel_err, total_ep_len = [], [], [], []

    for _ in range(num_episodes):
        obs, _ = env.reset()
        done = False
        ep_reward = 0.0
        ep_actions, ep_errors, steps = [], [], 0

        while not done:
            action, _ = agent.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_reward += reward
            ep_actions.append(np.linalg.norm(action))
            ep_errors.append(np.linalg.norm(obs[:3]))
            done = terminated or truncated
            steps += 1

        total_rewards.append(ep_reward)
        total_action_mag.append(np.mean(ep_actions))
        total_ang_vel_err.append(np.mean(ep_errors))
        total_ep_len.append(steps)

    avg_reward = float(np.mean(total_rewards))
    avg_action_mag = float(np.mean(total_action_mag))
    avg_ang_vel_err = float(np.mean(total_ang_vel_err))
    avg_ep_len = float(np.mean(total_ep_len))

    def _axis_track(axis, mag=2.0):
        sp = np.zeros(3, dtype=np.float32)
        sp[axis] = mag
        obs, _ = env.reset(options={"eval_setpoint": sp})
        errs, done = [], False
        while not done:
            action, _ = agent.predict(obs, deterministic=True)
            obs, _, term, trunc, _ = env.step(action)
            errs.append(abs(float(obs[axis]) - mag))  # |rate - sp| on that axis
            done = term or trunc
        tail = errs[len(errs) // 4 :] if len(errs) > 8 else errs
        return float(np.sqrt(np.mean(np.square(tail)))) if tail else float("nan")

    try:
        e_roll = _axis_track(0)
        e_pitch = _axis_track(1)
        parity = e_roll / e_pitch if e_pitch > 1e-6 else float("nan")
        logger.info(
            f"[PARITY] roll_err={e_roll:.3f}  pitch_err={e_pitch:.3f}  "
            f"roll/pitch={parity:.2f}  (target 1.00 = symmetric)"
        )
    except Exception as _e:
        logger.info(f"[PARITY] probe skipped ({_e})")

    logger.info(
        f"[EVAL] Reward={avg_reward:.2f}  ActionMag={avg_action_mag:.2f}  "
        f"TrackErr={avg_ang_vel_err:.2f}  EpLen={avg_ep_len:.1f}"
    )
    return avg_reward, avg_action_mag, avg_ang_vel_err, avg_ep_len


# Main training loop
def train_model(
    env,
    episodes: int = TOTAL_EPISODES,
    max_steps: int = 400,
    gamma: float = GAMMA,
    learning_rate: float = LEARNING_RATE,
    batch_size: int = BATCH_SIZE,
    n_epochs: int = N_EPOCHS,
    horizon: int = N_STEPS,
    model_save_path: str = f"/opt/fleet_model_ws/models/{MODEL_BASENAME}.zip",
):
    # v9: automatically detect obs_dim from the env. fleet_model_world v9 uses 12 (added
    # clamped error integrator). Probing prevents silent shape mismatches.
    _probe = gym.make("FleetModel-v0")
    OBS_DIM = int(np.prod(_probe.observation_space.shape))
    try:
        _probe.close()
    except Exception:
        pass
    logger.info(f"[PPO] Detected obs_dim = {OBS_DIM} from env")

    # Build or load agent
    if START_TRAINING_FROM == 0:
        agent = PPOAgent(
            obs_dim=OBS_DIM,
            act_dim=3,
            hidden=(64, 64),
            learning_rate=learning_rate,
            gamma=gamma,
            batch_size=batch_size,
            n_epochs=n_epochs,
            n_steps=horizon,
            ent_coef=0.00004,
        )
        logger.info("[PPO] Starting training from scratch.")
    else:
        ckpt_path = f"/opt/fleet_model_ws/models/{MODEL_BASENAME}_{START_TRAINING_FROM}"
        agent = PPOAgent.load(ckpt_path, learning_rate=learning_rate)
        logger.info(f"[PPO] Resuming from checkpoint: {ckpt_path}")

    # VecEnv
    vec_env = VecEnv(NUM_ENVS, lambda: gym.make("FleetModel-v0"))
    logger.info(f"[VecEnv] {NUM_ENVS} parallel envs created")

    # Open rollout log
    _open_rollout_log()
    logger.info(f"[LOG] Rollout log: {ROLLOUT_LOG_PATH}")

    # TensorBoard
    tb_writer = tf.summary.create_file_writer("./ppo_tensorboard/")

    # Metric histories
    reward_history, action_mag_history = [], []
    ang_vel_err_history, ep_len_history = [], []
    episode_numbers = []

    # Live plot
    plt.ion()
    fig, axs = plt.subplots(4, 1, figsize=(10, 12))
    axs[0].set_title("Training Progress: PPO (TF2)")
    labels = [
        "Average Reward",
        "Mean Action Magnitude",
        "Mean Tracking Error",
        "Steps per Episode",
    ]
    for ax, lbl in zip(axs, labels):
        ax.set_ylabel(lbl)
    axs[3].set_xlabel("Episodes")

    # Training loop
    timesteps_per_interval = EVAL_INTERVAL_EPISODES * max_steps

    for episode in range(0, episodes, EVAL_INTERVAL_EPISODES):

        interval_start = time.time()
        steps_this_interval = 0

        while steps_this_interval < timesteps_per_interval:

            _, ep_rewards, ep_lengths = agent.collect_rollout_vec(vec_env)

            steps_this_interval += agent.n_steps
            losses = agent.update()

            # Write per rollout diagnostic row
            _log_rollout(agent, losses, ep_rewards, ep_lengths)

            with tb_writer.as_default():
                step = agent.num_timesteps
                for key, val in losses.items():
                    tf.summary.scalar(key, val, step=step)
            tb_writer.flush()

            logger.info(
                f"[ROLLOUT] steps={steps_this_interval}/{timesteps_per_interval}"
                f"  total={agent.num_timesteps}"
                f"  loss={losses['train/loss']:.4f}"
                f"  pg={losses['train/policy_loss']:.4f}"
                f"  vf={losses['train/value_loss']:.4f}"
                f"  ent={losses['train/entropy_loss']:.4f}"
            )

        # Evaluate
        avg_reward, avg_act_mag, avg_ang_err, avg_ep_len = evaluate_model(
            env, agent, num_episodes=10
        )

        reward_history.append(avg_reward)
        action_mag_history.append(avg_act_mag)
        ang_vel_err_history.append(avg_ang_err)
        ep_len_history.append(avg_ep_len)
        episode_numbers.append(episode + EVAL_INTERVAL_EPISODES)

        with tb_writer.as_default():
            step = agent.num_timesteps
            tf.summary.scalar("eval/avg_reward", avg_reward, step=step)
            tf.summary.scalar("eval/avg_action_magnitude", avg_act_mag, step=step)
            tf.summary.scalar("eval/avg_tracking_error", avg_ang_err, step=step)
            tf.summary.scalar("eval/avg_episode_length", avg_ep_len, step=step)
        tb_writer.flush()

        # Live plot
        data_series = [
            reward_history,
            action_mag_history,
            ang_vel_err_history,
            ep_len_history,
        ]
        for i, (ax, data, lbl) in enumerate(zip(axs, data_series, labels)):
            ax.clear()
            ax.plot(episode_numbers, data, marker="o", linestyle="-")
            ax.set_ylabel(lbl)
            if i == 3:
                ax.set_xlabel("Episodes")
        plt.pause(0.1)

        # Timing
        interval_duration = time.time() - interval_start
        logger.info(
            f"TIME for {EVAL_INTERVAL_EPISODES} episodes = "
            f"{interval_duration:.2f} s  |  total_steps = {agent.num_timesteps}"
        )

        # Checkpoint & training_log.csv
        ckpt = (
            f"/opt/fleet_model_ws/models/"
            f"{MODEL_BASENAME}_{START_TRAINING_FROM + episode + EVAL_INTERVAL_EPISODES}"
        )
        agent.save(ckpt)
        logger.info(f"[PPO] Checkpoint saved: {ckpt}")

        write_header = not os.path.exists(TRAINING_LOG_PATH)
        obs_zero = tf.zeros((1, agent.obs_dim), dtype=tf.float32)  # v9: was hardcoded 9
        mean_zero, std_zero = agent.actor(obs_zero)

        with open(TRAINING_LOG_PATH, "a", newline="") as f:
            w = csv.writer(f)
            if write_header:
                w.writerow(
                    [
                        "episode",
                        "log_std_0",
                        "log_std_1",
                        "log_std_2",
                        "mean_0",
                        "mean_1",
                        "mean_2",
                        "avg_reward",
                        "avg_ep_len",
                    ]
                )
            w.writerow(
                [
                    START_TRAINING_FROM + episode + EVAL_INTERVAL_EPISODES,
                    round(float(agent.actor.log_std.numpy()[0]), 6),
                    round(float(agent.actor.log_std.numpy()[1]), 6),
                    round(float(agent.actor.log_std.numpy()[2]), 6),
                    round(float(mean_zero.numpy()[0][0]), 6),
                    round(float(mean_zero.numpy()[0][1]), 6),
                    round(float(mean_zero.numpy()[0][2]), 6),
                    round(avg_reward, 4),
                    round(avg_ep_len, 1),
                ]
            )

    # Final save
    final_path = model_save_path.replace(".zip", "")
    agent.save(final_path)
    logger.info(f"[PPO] Final model saved: {final_path}")

    if _rollout_log_file:
        _rollout_log_file.close()

    plt.ioff()
    plt.show()


# Entry point
def main(args=None):
    logger.info("[DEBUG] starting")

    logger.info("[DEBUG] calling gym.make...")
    env = gym.make("FleetModel-v0")
    logger.info("[DEBUG] env created")
    logger.info("Starting PPO training (TF2/Keras)")

    try:
        max_steps = env.spec.max_episode_steps  # 400
        start_time = time.time()
        train_model(env, episodes=TOTAL_EPISODES, max_steps=max_steps)
        logger.info(f"TOTAL Training time: {time.time() - start_time:.2f} s")

    except Exception as e:
        logger.info(f"Error during training: {e}")
        raise

    finally:
        if _rollout_log_file:
            try:
                _rollout_log_file.close()
            except Exception:
                pass
        # Flush any remaining step log rows to disk (FM only, no op otherwise)
        flush_step_log()
        if hasattr(env.unwrapped, "main_node"):
            env.unwrapped.main_node.destroy_node()
        env.close()


if __name__ == "__main__":
    main()
