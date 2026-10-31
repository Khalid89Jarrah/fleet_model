#!/usr/bin/env python3

import numpy as np
from pathlib import Path
import yaml
import logging

from sensor_interaction.psude_inverse import PseudoInverse
from sensor_interaction.sequentialdesaturationmotoroutput import (
    SequentialDesaturationMotorOutput,
)


class _PyLogger:
    def __init__(self):
        self._log = logging.getLogger("fast_drone_node")
        if not self._log.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(
                logging.Formatter("[%(name)s] %(levelname)s: %(message)s")
            )
            self._log.addHandler(handler)
            self._log.setLevel(logging.INFO)

    def info(self, msg):
        self._log.info(msg)

    def warning(self, msg):
        self._log.warning(msg)

    def warn(self, msg):
        self._log.warning(msg)

    def error(self, msg):
        self._log.error(msg)

    def debug(self, msg):
        self._log.debug(msg)


class _ImuMsg:
    # Mimic sensor_msgs/Imu shape fleet_model_world.py reads.

    class _Vec3:
        __slots__ = ("x", "y", "z")

        def __init__(self, x=0.0, y=0.0, z=0.0):
            self.x = float(x)
            self.y = float(y)
            self.z = float(z)

    class _Quat:
        __slots__ = ("x", "y", "z", "w")

        def __init__(self, x=0.0, y=0.0, z=0.0, w=1.0):
            self.x = float(x)
            self.y = float(y)
            self.z = float(z)
            self.w = float(w)

    def __init__(self, omega_noisy: np.ndarray, quat: np.ndarray):
        self.angular_velocity = self._Vec3(*omega_noisy)
        self.orientation = self._Quat(*quat)
        self.linear_acceleration = self._Vec3(0.0, 0.0, 0.0)


# Simulated drone node in the FRD body frame
class FastDroneNode:

    # Time step
    DT = 0.004
    EPISODE_S = 20.0

    # Mass
    MASS = 2.0
    IXX = 0.02166666666666667
    IYY = 0.02166666666666667
    IZZ = 0.04000000000000001

    # Rotor geometry, FRD body frame
    ROTOR_POS = np.array(
        [
            [+0.159, +0.159, -0.06],  # rotor 0 (front right, ccw)
            [-0.159, -0.159, -0.06],  # rotor 1 (rear left,   ccw)
            [+0.159, -0.159, -0.06],  # rotor 2 (front left,  cw)
            [-0.159, +0.159, -0.06],  # rotor 3 (rear right,  cw)
        ],
        dtype=np.float64,
    )

    # Aerodynamic constants
    KT = 8.54858e-06
    KM = 0.016
    TAU_UP = 0.0125
    TAU_DOWN = 0.025
    MAX_OMEGA = 1000.0
    PWM_MIN = 150.0
    PWM_MAX = 1000.0

    # Spin direction per rotor (sign of reaction torque on body)
    ROTOR_DIR = np.array([+1.0, +1.0, -1.0, -1.0], dtype=np.float64)

    # Thrust direction in FRD (rotors push body upward, so -z FRD)
    THRUST_AXIS = np.array([0.0, 0.0, -1.0], dtype=np.float64)

    # IMU noise
    IMU_WHITE_STD = 0.00018665
    IMU_BIAS_STD = 3.8785e-05
    IMU_BIAS_TAU = 1000.0

    # Per episode randomization ranges
    DR_KT_RANGE = (0.60, 1.40)
    DR_TAU_UP_RANGE = (1.00, 4.00)  # scales TAU_UP and TAU_DOWN together

    # Observation delay range in ticks of DT
    DR_OBS_DELAY_TICKS_RANGE = (6, 11)

    DR_INERTIA_RANGE = (0.85, 1.15)
    DR_IMU_NOISE_RANGE = (0.50, 1.50)

    # Gaussian turbulence torque std (max)
    DR_DISTURBANCE_MAX = 0.080

    # Sinusoidal gust amplitude and frequency limits
    DR_GUST_AMP_MAX = 0.200
    DR_GUST_FREQ_MIN = 0.05
    DR_GUST_FREQ_MAX = 2.0

    # Constant wind torque buckets: (min, max, probability)
    DR_WIND_BIAS_BUCKETS = (
        (0.000, 0.010, 0.25),  # calm     <2 m/s
        (0.010, 0.050, 0.25),  # light    ~5 m/s
        (0.050, 0.150, 0.20),  # moderate ~10 m/s
        (0.150, 0.300, 0.18),  # harsh    ~15 m/s
        (0.300, 0.390, 0.12),  # violent  ~20 m/s
    )

    # Wind direction shift: probability and time window (s) for windy episodes
    DR_WIND_SHIFT_PROB = 0.30
    DR_WIND_SHIFT_T_MIN = 5.0
    DR_WIND_SHIFT_T_MAX = 15.0

    # Per axis disturbance weight (equal on roll, pitch, yaw)
    DR_AXIS_DISTURBANCE_WEIGHT = np.array([1.0, 1.0, 1.0], dtype=np.float64)

    # Single motor failure probability and thrust multiplier buckets
    DR_MOTOR_FAIL_PROB = 0.50
    DR_MOTOR_FAIL_BUCKETS = (
        (0.90, 0.95, 0.20),  # mild
        (0.80, 0.90, 0.25),  # moderate
        (0.70, 0.80, 0.35),  # severe (the K=0.7 cliff)
        (0.60, 0.70, 0.20),  # extreme
    )
    DR_MOTOR_FAIL_MULT_MIN = 0.60
    DR_MOTOR_FAIL_MULT_MAX = 0.95

    # Chopped propeller: thrust scales with r**4 and drag with r**5
    DR_PROP_CHOP_PROB = 0.25  # share of damage episodes that are chops
    DR_PROP_CHOP_RADIUS = (0.85, 1.00)  # 0 to 15% radius loss

    # Mid flight damage onset probability and time window (s)
    DR_DAMAGE_MID_ONSET_PROB = 0.50
    DR_DAMAGE_ONSET_T_MIN = 3.0
    DR_DAMAGE_ONSET_T_MAX = 12.0

    # Battery sag: global thrust scale range per episode
    DR_BATTERY_SCALE_RANGE = (0.85, 1.00)

    # IMU vibration: peak amplitude, frequency band and number of tones
    DR_IMU_VIB_AMP_MAX = 0.50  # rad/s peak
    DR_IMU_VIB_FREQ_MIN = 15.0  # Hz
    DR_IMU_VIB_FREQ_MAX = 124.0  # Hz (below Nyquist = 125)
    DR_IMU_VIB_N_TONES = 3

    def __init__(self):
        self._logger = _PyLogger()
        self.timing_log = []

        self.psude_inverse = PseudoInverse()
        self.sequential_desaturation = SequentialDesaturationMotorOutput()

        current_dir = Path(__file__).resolve().parent
        yaml_path = current_dir.parent / "resources" / "fleet_model_drone" / "f450.yaml"
        with open(yaml_path, "r") as f:
            self.params = yaml.safe_load(f)

        self._I_nominal = np.diag([self.IXX, self.IYY, self.IZZ])
        self._I_INV_nominal = np.linalg.inv(self._I_nominal)

        # Physics state (FRD)
        self._omega = np.zeros(3, dtype=np.float64)
        self._quat = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
        self._rotor_omegas = np.zeros(4, dtype=np.float64)
        self._pwm_cmd = np.zeros(4, dtype=np.float64)
        self._sim_time = 0.0
        self._imu_bias = np.zeros(3, dtype=np.float64)

        # Per episode DR values
        self._kt_per_motor = np.full(4, self.KT, dtype=np.float64)
        self._tau_up_per_motor = np.full(4, self.TAU_UP, dtype=np.float64)
        self._tau_down_per_motor = np.full(4, self.TAU_DOWN, dtype=np.float64)
        self._I_episode = self._I_nominal.copy()
        self._I_INV_episode = self._I_INV_nominal.copy()
        self._disturbance_std = 0.0
        self._imu_noise_scale = 1.0
        self._OBS_DELAY_TICKS = self.DR_OBS_DELAY_TICKS_RANGE[0]
        self._obs_delay_buf = []

        # Gust profile
        self._gust_amp = np.zeros(3, dtype=np.float64)
        self._gust_freq = 0.0
        self._gust_phase = np.zeros(3, dtype=np.float64)

        # Constant wind bias
        self._wind_bias = np.zeros(3, dtype=np.float64)

        # Wind direction shift
        self._wind_bias_post = np.zeros(3, dtype=np.float64)
        self._wind_shift_t = 0.0
        self._wind_shifted = False

        # Motor failure and mid flight onset
        self._motor_fail_mult = np.ones(4, dtype=np.float64)
        self._damage_target_mult = np.ones(4, dtype=np.float64)
        self._damage_onset_t = 0.0
        self._damage_applied = True
        # Per motor drag multiplier, separate from the thrust multiplier
        self._damage_km_mult = np.ones(4, dtype=np.float64)
        self._damage_km_target_mult = np.ones(4, dtype=np.float64)
        # True only on the tick damage becomes active
        self.damage_just_applied = False

        # Battery sag
        self._battery_scale = 1.0

        # IMU vibration (tones x axes)
        _n_vib = self.DR_IMU_VIB_N_TONES
        self._imu_vib_amp = np.zeros((_n_vib, 3), dtype=np.float64)
        self._imu_vib_freq = np.zeros(_n_vib, dtype=np.float64)  # Hz
        self._imu_vib_phase = np.zeros((_n_vib, 3), dtype=np.float64)

        self._logger.info("[FastDroneNode] FRD numpy sim")
        self._logger.info(
            "[FastDroneNode] DR: wind to 20 m/s + dir shifts + damage onset + battery sag + IMU vibration"
        )

    # Public interface

    def get_logger(self):
        return self._logger

    def reset_simulation(self, hover_pwm=None):
        # Reset physics state
        self._omega = np.zeros(3, dtype=np.float64)
        self._quat = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
        self._rotor_omegas = np.zeros(4, dtype=np.float64)
        self._pwm_cmd = np.zeros(4, dtype=np.float64)
        self._sim_time = 0.0
        self._imu_bias = np.zeros(3, dtype=np.float64)
        self._obs_delay_buf = []

        if hover_pwm is not None:
            pwm = np.array(hover_pwm, dtype=np.float64)
            self._rotor_omegas = pwm.copy()
            self._pwm_cmd = pwm.copy()

        # Physical DR
        self._kt_per_motor = self.KT * np.random.uniform(*self.DR_KT_RANGE, 4)
        tau_scale = np.random.uniform(*self.DR_TAU_UP_RANGE, 4)
        self._tau_up_per_motor = self.TAU_UP * tau_scale
        self._tau_down_per_motor = self.TAU_DOWN * tau_scale
        inertia_scale = np.random.uniform(*self.DR_INERTIA_RANGE)
        self._I_episode = self._I_nominal * inertia_scale
        self._I_INV_episode = self._I_INV_nominal / inertia_scale
        self._disturbance_std = np.random.uniform(0.0, self.DR_DISTURBANCE_MAX)
        self._imu_noise_scale = np.random.uniform(*self.DR_IMU_NOISE_RANGE)
        lo_d, hi_d = self.DR_OBS_DELAY_TICKS_RANGE
        self._OBS_DELAY_TICKS = int(np.random.randint(lo_d, hi_d + 1))

        # Per episode gust profile
        self._gust_freq = float(
            np.random.uniform(self.DR_GUST_FREQ_MIN, self.DR_GUST_FREQ_MAX)
        )
        self._gust_phase = np.random.uniform(0.0, 2.0 * np.pi, 3)
        gust_scale = np.random.uniform(0.0, self.DR_GUST_AMP_MAX)
        self._gust_amp = gust_scale * np.random.uniform(0.3, 1.0, 3)

        # Constant wind bias with optional direction shift
        bp = np.array([b[2] for b in self.DR_WIND_BIAS_BUCKETS])
        wbi = int(np.random.choice(len(self.DR_WIND_BIAS_BUCKETS), p=bp))
        wlo, whi, _ = self.DR_WIND_BIAS_BUCKETS[wbi]
        wmag = float(np.random.uniform(wlo, whi))
        wdir = _random_unit_vec3()
        self._wind_bias = wmag * wdir

        if wbi >= 1 and np.random.random() < self.DR_WIND_SHIFT_PROB:
            wdir2 = _random_unit_vec3()
            self._wind_bias_post = wmag * wdir2
            self._wind_shift_t = float(
                np.random.uniform(self.DR_WIND_SHIFT_T_MIN, self.DR_WIND_SHIFT_T_MAX)
            )
        else:
            self._wind_bias_post = self._wind_bias.copy()
            self._wind_shift_t = 0.0
        self._wind_shifted = False

        # Motor failure with optional mid flight onset
        self._motor_fail_mult = np.ones(4, dtype=np.float64)
        self._damage_target_mult = np.ones(4, dtype=np.float64)
        self._damage_km_mult = np.ones(4, dtype=np.float64)
        self._damage_km_target_mult = np.ones(4, dtype=np.float64)
        self._damage_onset_t = 0.0
        self._damage_applied = True
        self.damage_just_applied = False

        if np.random.random() < self.DR_MOTOR_FAIL_PROB:
            failed_idx = int(np.random.randint(0, 4))

            if np.random.random() < self.DR_PROP_CHOP_PROB:
                # Chopped blade: thrust ~ r^4, drag ~ r^5
                r = float(np.random.uniform(*self.DR_PROP_CHOP_RADIUS))
                self._damage_target_mult[failed_idx] = r**4
                self._damage_km_target_mult[failed_idx] = r**5
            else:
                # Weak motor / ESC: thrust and drag fall together.
                mp = np.array([b[2] for b in self.DR_MOTOR_FAIL_BUCKETS])
                mbi = int(np.random.choice(len(self.DR_MOTOR_FAIL_BUCKETS), p=mp))
                mlo, mhi, _ = self.DR_MOTOR_FAIL_BUCKETS[mbi]
                mult = float(np.random.uniform(mlo, mhi))
                self._damage_target_mult[failed_idx] = mult
                self._damage_km_target_mult[failed_idx] = mult

            if np.random.random() < self.DR_DAMAGE_MID_ONSET_PROB:
                self._damage_onset_t = float(
                    np.random.uniform(
                        self.DR_DAMAGE_ONSET_T_MIN, self.DR_DAMAGE_ONSET_T_MAX
                    )
                )
                self._damage_applied = False
            else:
                self._motor_fail_mult = self._damage_target_mult.copy()
                self._damage_km_mult = self._damage_km_target_mult.copy()
                self._damage_applied = True

        # Battery sag: global thrust scale for the episode
        self._battery_scale = float(np.random.uniform(*self.DR_BATTERY_SCALE_RANGE))

        # IMU vibration: random tone frequencies, phases and amplitudes
        _n_vib = self.DR_IMU_VIB_N_TONES
        self._imu_vib_freq = np.random.uniform(
            self.DR_IMU_VIB_FREQ_MIN, self.DR_IMU_VIB_FREQ_MAX, _n_vib
        )
        self._imu_vib_phase = np.random.uniform(0.0, 2.0 * np.pi, (_n_vib, 3))
        vib_scale = np.random.uniform(0.0, self.DR_IMU_VIB_AMP_MAX)
        # Per tone, per axis amplitude in [0.3, 1.0] x scale
        self._imu_vib_amp = vib_scale * np.random.uniform(0.3, 1.0, (_n_vib, 3))

    def publish_motor_commands(self, motor_commands):
        self._pwm_cmd = np.array(motor_commands, dtype=np.float64)

    def perform_simulation_step(self, steps: int = 1):
        for _ in range(steps):
            self._step_physics()

    def read_imu(self):
        # White noise (scaled per episode)
        white = np.random.normal(0.0, self.IMU_WHITE_STD * self._imu_noise_scale, 3)
        # Sum of sinusoidal vibration tones per axis
        vib = np.sum(
            self._imu_vib_amp
            * np.sin(
                2.0 * np.pi * self._imu_vib_freq[:, None] * self._sim_time
                + self._imu_vib_phase
            ),
            axis=0,
        )
        noisy_omega = self._omega + self._imu_bias + white + vib

        current = _ImuMsg(noisy_omega, self._quat)
        if self._OBS_DELAY_TICKS <= 0:
            return current
        self._obs_delay_buf.append(current)
        if len(self._obs_delay_buf) <= self._OBS_DELAY_TICKS:
            return self._obs_delay_buf[0]
        return self._obs_delay_buf.pop(0)

    def get_sim_time(self):
        return self._sim_time

    # Control allocation passthroughs (unchanged)

    def read_motor_specs(self):
        rotors = []
        for _, v in self.params["rotors"].items():
            rotors.append(
                {
                    "motor_constant": v["KM"],
                    "position": np.array([v["PX"], v["PY"], 0.0]),
                    "axis": np.array([v["AX"], v["AY"], v["AZ"]]),
                    "ct": v["CT"],
                }
            )
        return {"rotors": rotors}

    def compute_effectiveness_matrix(self, specs):
        rotors = specs["rotors"]
        M = np.zeros((6, len(rotors)), dtype=np.float64)
        for i, r in enumerate(rotors):
            axis = r["axis"].astype(np.float64)
            n = np.linalg.norm(axis)
            if n > np.finfo(float).eps:
                axis /= n
            pos = r["position"].astype(np.float64)
            ct, km = r["ct"], r["motor_constant"]
            M[:3, i] = ct * np.cross(pos, axis) - ct * km * axis
            M[3:, i] = ct * axis
        return M

    def compute_psudo_inverse(self, effectiveness_matrix):
        return self.psude_inverse.update_pseudo_inverse_full(effectiveness_matrix)

    def compute_sequential_desaturation(self, rate_control, psudo_inverse_matrix):
        return self.sequential_desaturation.mixAirmodeDisabled(
            rate_control, psudo_inverse_matrix
        )

    def destroy_node(self):
        pass

    def stop_executor(self):
        pass

    # Private physics
    def _step_physics(self):
        dt = self.DT

        # Apply mid flight damage once its onset time is reached
        self.damage_just_applied = False
        if (not self._damage_applied) and self._sim_time >= self._damage_onset_t:
            self._motor_fail_mult = self._damage_target_mult.copy()
            self._damage_km_mult = self._damage_km_target_mult.copy()
            self._damage_applied = True
            self.damage_just_applied = True

        # Switch to the post shift wind direction once its time is reached
        if (
            (self._wind_shift_t > 0.0)
            and (not self._wind_shifted)
            and self._sim_time >= self._wind_shift_t
        ):
            self._wind_bias = self._wind_bias_post
            self._wind_shifted = True

        # Motor first order lag
        omega_des = np.clip(self._pwm_cmd, 0.0, self.MAX_OMEGA)
        for i in range(4):
            tau = (
                self._tau_up_per_motor[i]
                if omega_des[i] >= self._rotor_omegas[i]
                else self._tau_down_per_motor[i]
            )
            self._rotor_omegas[i] += (omega_des[i] - self._rotor_omegas[i]) / tau * dt
            self._rotor_omegas[i] = np.clip(self._rotor_omegas[i], 0.0, self.MAX_OMEGA)

        # Body torques (FRD)
        omega_sq = self._rotor_omegas**2
        thrusts = self._kt_per_motor * omega_sq

        # Thrust scales with motor failure and battery sag
        thrusts *= self._motor_fail_mult * self._battery_scale

        tau_body = np.zeros(3, dtype=np.float64)
        for i in range(4):
            F_i = thrusts[i] * self.THRUST_AXIS
            tau_body += np.cross(self.ROTOR_POS[i], F_i)
            # Reaction torque uses the drag multiplier and battery scale
            tau_body[2] += (
                self.ROTOR_DIR[i]
                * self._kt_per_motor[i]
                * self.KM
                * omega_sq[i]
                * self._damage_km_mult[i]
                * self._battery_scale
            )

        # External disturbance: turbulence, gust and wind bias
        w = self.DR_AXIS_DISTURBANCE_WEIGHT

        if self._disturbance_std > 0.0:
            tau_body += w * np.random.normal(0.0, self._disturbance_std, 3)

        tau_body += (
            w
            * self._gust_amp
            * np.sin(2.0 * np.pi * self._gust_freq * self._sim_time + self._gust_phase)
        )
        tau_body += w * self._wind_bias

        # Newton Euler
        I_omega = self._I_episode @ self._omega
        alpha = self._I_INV_episode @ (tau_body - np.cross(self._omega, I_omega))

        # Integrate omega
        self._omega = self._omega + alpha * dt

        # Integrate quaternion (body frame angular velocity)
        x, y, z, w_q = self._quat
        wx, wy, wz = self._omega * dt * 0.5
        q = np.array(
            [
                x + w_q * wx + y * wz - z * wy,
                y + w_q * wy + z * wx - x * wz,
                z + w_q * wz + x * wy - y * wx,
                w_q - x * wx - y * wy - z * wz,
            ],
            dtype=np.float64,
        )
        n = np.linalg.norm(q)
        self._quat = q / n if n > 1e-10 else self._quat

        # IMU bias dynamics (Ornstein Uhlenbeck)
        sigma = self.IMU_BIAS_STD * np.sqrt(2.0 / self.IMU_BIAS_TAU)
        self._imu_bias += (
            -self._imu_bias / self.IMU_BIAS_TAU + np.random.normal(0.0, sigma, 3)
        ) * dt

        self._sim_time += dt


def _random_unit_vec3():
    # Uniform random unit vector on the three sphere.
    v = np.random.normal(0.0, 1.0, 3)
    n = np.linalg.norm(v)
    if n > 1e-9:
        return v / n
    return np.array([1.0, 0.0, 0.0])
