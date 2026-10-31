#!/usr/bin/env python3
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras

_gpus = tf.config.list_physical_devices("GPU")
if _gpus:
    try:
        tf.config.experimental.set_memory_growth(_gpus[0], True)
        tf.config.set_visible_devices(_gpus[0], "GPU")
    except RuntimeError:
        pass

DEVICE = "/GPU:0" if _gpus else "/CPU:0"

SIGN_LAMBDA = 1.0  # weight of the sign penalty (0.0 disables it)
SIGN_MARGIN = 0.10  # hinge floor for Ji = d(mean[a])/d(e_I[a])
SIGN_EI_IDX = [9, 10, 11]
SIGN_MARGIN_W = 0.03
SIGN_W_IDX = [0, 1, 2]


class VecEnv:
    def __init__(self, num_envs: int, env_factory):
        self.num_envs = num_envs
        self.envs = [env_factory() for _ in range(num_envs)]
        # Obs dim is read from the env
        obs_dim = int(np.prod(self.envs[0].observation_space.shape))
        self._obs = np.zeros((num_envs, obs_dim), dtype=np.float32)
        for i, env in enumerate(self.envs):
            obs, _ = env.reset()
            self._obs[i] = obs

    def step(self, actions: np.ndarray):
        obs_out = np.zeros_like(self._obs)
        rewards = np.zeros(self.num_envs, dtype=np.float32)
        dones = np.zeros(self.num_envs, dtype=bool)
        for i, env in enumerate(self.envs):
            next_obs, reward, terminated, truncated, _ = env.step(actions[i])
            done = terminated or truncated
            if done:
                next_obs, _ = env.reset()
            obs_out[i] = next_obs
            rewards[i] = reward
            dones[i] = done
        self._obs = obs_out
        return obs_out.copy(), rewards, dones

    def reset(self):
        for i, env in enumerate(self.envs):
            obs, _ = env.reset()
            self._obs[i] = obs
        return self._obs.copy()


class Actor(keras.Model):
    def __init__(self, obs_dim: int, act_dim: int, hidden: tuple = (64, 64)):
        super().__init__()
        self.act_dim = act_dim
        self.h0 = keras.layers.Dense(hidden[0], activation="tanh", name="actor_h1")
        self.h1 = keras.layers.Dense(hidden[1], activation="tanh", name="actor_h2")
        self.mean_head = keras.layers.Dense(act_dim, activation=None, name="actor_mean")

    def build(self, input_shape):
        self.log_std = self.add_weight(
            name="log_std",
            shape=(self.act_dim,),
            initializer=tf.keras.initializers.Constant(-1.0),
            trainable=True,
        )
        super().build(input_shape)

    def call(self, obs):
        x = self.h0(obs)
        x = self.h1(x)
        mean = self.mean_head(x)
        log_std = tf.clip_by_value(self.log_std, -2.5, -2.0)
        std = tf.exp(log_std)
        std = tf.broadcast_to(std, tf.shape(mean))
        return mean, std

    def get_action(self, obs, deterministic: bool = False):
        with tf.device(DEVICE):
            obs_t = tf.cast(obs, tf.float32)
            mean, std = self(obs_t)
            action_unclipped = (
                mean if deterministic else mean + std * tf.random.normal(tf.shape(mean))
            )
            # Log prob is taken on the unclipped action to keep it in a normal range
            log_prob = _gaussian_log_prob(action_unclipped, mean, std)
            action = tf.clip_by_value(action_unclipped, -1.0, 1.0)
        return action.numpy(), log_prob.numpy()


class Critic(keras.Model):
    def __init__(self, obs_dim: int, hidden: tuple = (64, 64)):
        super().__init__()
        self.h1 = keras.layers.Dense(hidden[0], activation="tanh", name="critic_h1")
        self.h2 = keras.layers.Dense(hidden[1], activation="tanh", name="critic_h2")
        self.value_head = keras.layers.Dense(1, activation=None, name="critic_value")

    def call(self, obs):
        x = self.h1(obs)
        x = self.h2(x)
        return self.value_head(x)


class RolloutBuffer:
    def __init__(self, n_steps, obs_dim, act_dim, gamma=0.99, gae_lambda=0.95):
        self.n_steps = n_steps
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.obs = np.zeros((n_steps, obs_dim), dtype=np.float32)
        self.actions = np.zeros((n_steps, act_dim), dtype=np.float32)
        self.rewards = np.zeros(n_steps, dtype=np.float32)
        self.dones = np.zeros(n_steps, dtype=np.float32)
        self.values = np.zeros(n_steps, dtype=np.float32)
        self.log_probs = np.zeros(n_steps, dtype=np.float32)
        self.returns = np.zeros(n_steps, dtype=np.float32)
        self.advantages = np.zeros(n_steps, dtype=np.float32)
        self.ptr = 0
        self.full = False

    def add(self, obs, action, reward, done, value, log_prob):
        self.obs[self.ptr] = obs
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = float(done)
        self.values[self.ptr] = value
        self.log_probs[self.ptr] = log_prob
        self.ptr += 1
        if self.ptr == self.n_steps:
            self.full = True

    def compute_returns_and_advantages_slice(
        self, start: int, end: int, last_value: float
    ):
        last_gae = 0.0
        for t in reversed(range(start, end)):
            nnt = 1.0 - float(self.dones[t])
            nv = last_value if t == end - 1 else self.values[t + 1]
            delta = self.rewards[t] + self.gamma * nv * nnt - self.values[t]
            last_gae = delta + self.gamma * self.gae_lambda * nnt * last_gae
            self.advantages[t] = last_gae
        self.returns[start:end] = self.advantages[start:end] + self.values[start:end]

    def to_gpu_tensors(self):
        with tf.device(DEVICE):
            return {
                "obs": tf.constant(self.obs),
                "actions": tf.constant(self.actions),
                "log_probs": tf.constant(self.log_probs),
                "advantages": tf.constant(self.advantages),
                "returns": tf.constant(self.returns),
            }

    def get_batches(self, tensors: dict, batch_size: int):
        indices = np.random.permutation(self.n_steps)
        for start in range(0, self.n_steps, batch_size):
            idx = tf.constant(indices[start : start + batch_size])
            yield (
                tf.gather(tensors["obs"], idx),
                tf.gather(tensors["actions"], idx),
                tf.gather(tensors["log_probs"], idx),
                tf.gather(tensors["advantages"], idx),
                tf.gather(tensors["returns"], idx),
            )

    def reset(self):
        self.ptr = 0
        self.full = False


# Separate gradient functions for actor and critic (each has its own tape and optimizer)
@tf.function
def _ppo_compute_actor_grads(
    actor,
    obs_b,
    act_b,
    old_log_b,
    adv_b,
    clip_range,
    ent_coef,
    max_grad_norm,
):
    # Compute actor loss and clipped gradients. Does NOT apply them.
    with tf.GradientTape() as tape:
        mean, std = actor(obs_b, training=True)
        # Stop gradient on std so log_std only receives the entropy gradient
        new_log_prob = _gaussian_log_prob(act_b, mean, tf.stop_gradient(std))
        entropy = _gaussian_entropy(std)

        ratio = tf.exp(new_log_prob - old_log_b)
        pg_loss1 = -adv_b * ratio
        pg_loss2 = -adv_b * tf.clip_by_value(ratio, 1.0 - clip_range, 1.0 + clip_range)
        pg_loss = tf.reduce_mean(tf.maximum(pg_loss1, pg_loss2))
        ent_loss = tf.reduce_mean(entropy)

        W0 = actor.h0.kernel
        W1 = actor.h1.kernel
        W2 = actor.mean_head.kernel
        obs_j = tf.concat([obs_b, tf.zeros_like(obs_b[:1])], axis=0)
        hj1 = tf.tanh(tf.matmul(obs_j, W0) + actor.h0.bias)
        hj2 = tf.tanh(tf.matmul(hj1, W1) + actor.h1.bias)
        dj1 = 1.0 - tf.square(hj1)  # (N, 64)
        dj2 = 1.0 - tf.square(hj2)  # (N, 64)
        Wi = tf.gather(W0, SIGN_EI_IDX, axis=0)  # (3, 64)
        tj1 = dj1[:, None, :] * Wi[None, :, :]  # (N, 3, 64)
        tj2 = tf.einsum("naj,jk->nak", tj1, W1) * dj2[:, None, :]  # (N, 3, 64)
        Ji = tf.einsum("nak,ka->na", tj2, W2)  # (N, 3)
        # One sided hinge: zero while Ji >= SIGN_MARGIN.
        sign_pen = tf.reduce_mean(tf.nn.relu(SIGN_MARGIN - Ji))
        # Same construction on the rate rows W0[0:3]; wanted sign is negative.
        Wr = tf.gather(W0, SIGN_W_IDX, axis=0)
        tr1 = dj1[:, None, :] * Wr[None, :, :]
        tr2 = tf.einsum("naj,jk->nak", tr1, W1) * dj2[:, None, :]
        Jw = tf.einsum("nak,ka->na", tr2, W2)
        sign_pen = sign_pen + tf.reduce_mean(tf.nn.relu(SIGN_MARGIN_W + Jw))

        actor_loss = pg_loss - ent_coef * ent_loss + SIGN_LAMBDA * sign_pen

    grads = tape.gradient(actor_loss, actor.trainable_variables)
    clipped_grads, _ = tf.clip_by_global_norm(grads, max_grad_norm)
    grad_norm = tf.linalg.global_norm(clipped_grads)
    return (
        pg_loss,
        ent_loss,
        actor_loss,
        clipped_grads,
        grad_norm,
        sign_pen,
        tf.reduce_mean(Ji, axis=0),
    )


@tf.function
def _ppo_compute_critic_grads(
    critic,
    obs_b,
    ret_b,
    vf_coef,
    max_grad_norm,
):
    # Compute critic loss and clipped gradients. Does NOT apply them.
    with tf.GradientTape() as tape:
        value_pred = tf.squeeze(critic(obs_b, training=True), axis=1)
        vf_loss = tf.reduce_mean(tf.square(ret_b - value_pred))
        critic_loss = vf_coef * vf_loss

    grads = tape.gradient(critic_loss, critic.trainable_variables)
    clipped_grads, _ = tf.clip_by_global_norm(grads, max_grad_norm)
    grad_norm = tf.linalg.global_norm(clipped_grads)
    return vf_loss, critic_loss, clipped_grads, grad_norm


class PPOAgent:
    def __init__(
        self,
        obs_dim: int = 9,
        act_dim: int = 3,
        hidden: tuple = (64, 64),
        learning_rate: float = 1e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: float = 0.2,
        n_epochs: int = 10,
        batch_size: int = 64,
        n_steps: int = 2048,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
    ):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.gamma = gamma
        self.clip_range = clip_range
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.n_steps = n_steps
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.num_timesteps = 0

        with tf.device(DEVICE):
            self.actor = Actor(obs_dim, act_dim, hidden)
            self.critic = Critic(obs_dim, hidden)

            # Separate actor and critic optimizers (critic lr is 5x higher)
            self.actor_optimizer = keras.optimizers.Adam(
                learning_rate=learning_rate, epsilon=1e-5
            )
            self.critic_optimizer = keras.optimizers.Adam(
                learning_rate=learning_rate * 5.0, epsilon=1e-5
            )

            dummy = tf.zeros((1, obs_dim))
            self.actor(dummy)
            self.critic(dummy)

        self.buffer = RolloutBuffer(n_steps, obs_dim, act_dim, gamma, gae_lambda)

    def predict(self, obs: np.ndarray, deterministic: bool = True):
        obs = np.array(obs, dtype=np.float32)
        if obs.ndim == 1:
            obs = obs[np.newaxis, :]
        action, _ = self.actor.get_action(obs, deterministic=deterministic)
        return action.squeeze(0), None

    def integral_sign(self, obs: np.ndarray = None):
        # Ji = d(tau_axis)/d(e_I_axis) per axis (roll, pitch, yaw); defaults to the zero observation.
        if obs is None:
            obs = np.zeros((1, self.obs_dim), dtype=np.float32)
        obs = np.atleast_2d(np.asarray(obs, dtype=np.float32))
        W0 = self.actor.h0.kernel.numpy()
        W1 = self.actor.h1.kernel.numpy()
        W2 = self.actor.mean_head.kernel.numpy()
        b0 = self.actor.h0.bias.numpy()
        b1 = self.actor.h1.bias.numpy()
        h1 = np.tanh(obs @ W0 + b0)
        h2 = np.tanh(h1 @ W1 + b1)
        d1 = 1.0 - h1**2
        d2 = 1.0 - h2**2
        Wi = W0[SIGN_EI_IDX, :]
        t1 = d1[:, None, :] * Wi[None, :, :]
        t2 = np.einsum("naj,jk->nak", t1, W1) * d2[:, None, :]
        return np.einsum("nak,ka->na", t2, W2).mean(axis=0)

    def collect_rollout_vec(self, vec_env: VecEnv):
        self.buffer.reset()
        num_envs = vec_env.num_envs
        steps_per_env = self.n_steps // num_envs
        obs_batch = vec_env.reset()

        all_obs = np.zeros((steps_per_env, num_envs, self.obs_dim), dtype=np.float32)
        all_actions = np.zeros(
            (steps_per_env, num_envs, self.act_dim), dtype=np.float32
        )
        all_rewards = np.zeros((steps_per_env, num_envs), dtype=np.float32)
        all_dones = np.zeros((steps_per_env, num_envs), dtype=np.float32)
        all_values = np.zeros((steps_per_env, num_envs), dtype=np.float32)
        all_log_probs = np.zeros((steps_per_env, num_envs), dtype=np.float32)

        ep_rewards, ep_lengths = [], []
        ep_rew = np.zeros(num_envs, dtype=np.float32)
        ep_len = np.zeros(num_envs, dtype=np.int32)

        for t in range(steps_per_env):
            with tf.device(DEVICE):
                obs_t = tf.constant(obs_batch, dtype=tf.float32)
                actions, log_probs = self.actor.get_action(obs_t)
                values = tf.squeeze(self.critic(obs_t), axis=1).numpy()

            # Replace NaN/Inf actions with 0 before stepping the env
            actions = np.where(np.isnan(actions), 0.0, actions)
            actions = np.where(np.isinf(actions), 0.0, actions)
            actions = np.clip(actions, -1.0, 1.0)

            next_obs_batch, rewards, dones = vec_env.step(actions)

            all_obs[t] = obs_batch
            all_actions[t] = actions
            all_rewards[t] = rewards
            all_dones[t] = dones.astype(np.float32)
            all_values[t] = values
            all_log_probs[t] = log_probs
            self.num_timesteps += num_envs

            ep_rew += rewards
            ep_len += 1
            for i in np.where(dones)[0]:
                ep_rewards.append(float(ep_rew[i]))
                ep_lengths.append(int(ep_len[i]))
                ep_rew[i] = 0.0
                ep_len[i] = 0

            obs_batch = next_obs_batch

        for i in range(num_envs):
            start = i * steps_per_env
            end = start + steps_per_env
            self.buffer.obs[start:end] = all_obs[:, i, :]
            self.buffer.actions[start:end] = all_actions[:, i, :]
            self.buffer.rewards[start:end] = all_rewards[:, i]
            self.buffer.dones[start:end] = all_dones[:, i]
            self.buffer.values[start:end] = all_values[:, i]
            self.buffer.log_probs[start:end] = all_log_probs[:, i]

        self.buffer.ptr = self.n_steps
        self.buffer.full = True

        # Normalise rewards to mean 0, std 1 before GAE
        r_mean = self.buffer.rewards.mean()
        r_std = self.buffer.rewards.std() + 1e-8
        self.buffer.rewards = (self.buffer.rewards - r_mean) / r_std

        # Bootstrap each env with its own last value
        with tf.device(DEVICE):
            last_values = tf.squeeze(
                self.critic(tf.constant(obs_batch, dtype=tf.float32)), axis=1
            ).numpy()

        for i in range(num_envs):
            start = i * steps_per_env
            end = start + steps_per_env
            self.buffer.compute_returns_and_advantages_slice(
                start, end, float(last_values[i])
            )

        return obs_batch, ep_rewards, ep_lengths

    def update(self):
        adv = self.buffer.advantages
        self.buffer.advantages = (adv - adv.mean()) / (adv.std() + 1e-8)
        tensors = self.buffer.to_gpu_tensors()

        total_pg, total_vf, total_ent, total = 0.0, 0.0, 0.0, 0.0
        # Sign penalty diagnostics
        total_sgn = 0.0
        total_ji = np.zeros(3, dtype=np.float64)
        n_updates = 0
        nan_skips = 0

        clip_range = tf.constant(self.clip_range, dtype=tf.float32)
        vf_coef = tf.constant(self.vf_coef, dtype=tf.float32)
        ent_coef = tf.constant(self.ent_coef, dtype=tf.float32)
        max_grad_norm = tf.constant(self.max_grad_norm, dtype=tf.float32)

        with tf.device(DEVICE):
            for _ in range(self.n_epochs):
                for obs_b, act_b, old_log_b, adv_b, ret_b in self.buffer.get_batches(
                    tensors, self.batch_size
                ):
                    # Actor step
                    pg, ent, a_loss, a_grads, a_gnorm, sgn_pen, sgn_ji = (
                        _ppo_compute_actor_grads(
                            self.actor,
                            obs_b,
                            act_b,
                            old_log_b,
                            adv_b,
                            clip_range,
                            ent_coef,
                            max_grad_norm,
                        )
                    )

                    # Skip the update if the actor gradients are NaN/Inf
                    if (
                        np.isnan(a_gnorm)
                        or np.isinf(a_gnorm)
                        or np.isnan(a_loss.numpy())
                    ):
                        nan_skips += 1
                        continue

                    self.actor_optimizer.apply_gradients(
                        zip(a_grads, self.actor.trainable_variables)
                    )

                    # Critic step
                    vf, c_loss, c_grads, c_gnorm = _ppo_compute_critic_grads(
                        self.critic,
                        obs_b,
                        ret_b,
                        vf_coef,
                        max_grad_norm,
                    )

                    # Skip the update if the critic gradients are NaN/Inf
                    if (
                        np.isnan(c_gnorm)
                        or np.isinf(c_gnorm)
                        or np.isnan(c_loss.numpy())
                    ):
                        nan_skips += 1
                        continue

                    self.critic_optimizer.apply_gradients(
                        zip(c_grads, self.critic.trainable_variables)
                    )

                    total_pg += pg.numpy()
                    total_vf += vf.numpy()
                    total_ent += ent.numpy()
                    total += a_loss.numpy() + c_loss.numpy()
                    total_sgn += sgn_pen.numpy()
                    total_ji += sgn_ji.numpy()
                    n_updates += 1

        if n_updates == 0:
            return {
                "train/policy_loss": 0.0,
                "train/value_loss": 0.0,
                "train/entropy_loss": 0.0,
                "train/loss": 0.0,
                "train/sign_penalty": 0.0,
                "train/Ji_roll": 0.0,
                "train/Ji_pitch": 0.0,
                "train/Ji_yaw": 0.0,
            }

        return {
            "train/policy_loss": total_pg / n_updates,
            "train/value_loss": total_vf / n_updates,
            "train/entropy_loss": total_ent / n_updates,
            "train/loss": total / n_updates,
            # Sign penalty and mean Ji per axis (Ji should be positive)
            "train/sign_penalty": total_sgn / n_updates,
            "train/Ji_roll": total_ji[0] / n_updates,
            "train/Ji_pitch": total_ji[1] / n_updates,
            "train/Ji_yaw": total_ji[2] / n_updates,
        }

    def save(self, path: str):
        os.makedirs(
            os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True
        )
        with tf.device(DEVICE):
            dummy = tf.zeros((1, self.obs_dim))
            self.actor(dummy)
            self.critic(dummy)
        self.actor.save_weights(f"{path}_actor.weights.h5")
        self.critic.save_weights(f"{path}_critic.weights.h5")
        config = {
            "obs_dim": self.obs_dim,
            "act_dim": self.act_dim,
            "n_steps": self.n_steps,
            "batch_size": self.batch_size,
            "n_epochs": self.n_epochs,
            "clip_range": self.clip_range,
            "ent_coef": self.ent_coef,
            "vf_coef": self.vf_coef,
            "max_grad_norm": self.max_grad_norm,
        }
        np.save(f"{path}_config.npy", config)
        ji = self.integral_sign()
        print(
            f"[PPO] Saved -> {path}_actor.weights.h5 / _critic.weights.h5 / _config.npy"
        )
        print(
            f"[PPO] Ji(e_I) roll={ji[0]:+.5f} pitch={ji[1]:+.5f} yaw={ji[2]:+.5f}"
            f"  {'OK' if np.all(ji > 0.0) else 'WRONG SIGN'}"
        )

    @classmethod
    def load(cls, path: str, learning_rate: float = 1e-4, **kwargs):
        config = np.load(f"{path}_config.npy", allow_pickle=True).item()
        config["learning_rate"] = learning_rate
        config.update(kwargs)
        agent = cls(**config)
        with tf.device(DEVICE):
            dummy = tf.zeros((1, config["obs_dim"]))
            agent.actor(dummy)
            agent.critic(dummy)
        agent.actor.load_weights(f"{path}_actor.weights.h5")
        agent.critic.load_weights(f"{path}_critic.weights.h5")
        print(f"[PPO] Loaded <- {path}_actor.weights.h5 / _critic.weights.h5")
        return agent


def _gaussian_log_prob(actions, mean, std):
    log_prob = (
        -0.5 * tf.square((actions - mean) / (std + 1e-8))
        - tf.math.log(std + 1e-8)
        - 0.5 * tf.math.log(2.0 * np.pi)
    )
    return tf.reduce_sum(log_prob, axis=-1)


def _gaussian_entropy(std):
    entropy = 0.5 + 0.5 * tf.math.log(2.0 * np.pi) + tf.math.log(std + 1e-8)
    return tf.reduce_sum(entropy, axis=-1)
