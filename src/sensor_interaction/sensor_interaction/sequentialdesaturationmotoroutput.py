# Portions derived from PX4, BSD 3 Clause. See THIRD_PARTY_LICENSES.
import numpy as np


class SequentialDesaturationMotorOutput:
    def __init__(self):
        self._num_actuators = 4
        self._actuator_min = np.array([0.0, 0.0, 0.0, 0.0])
        self._actuator_max = np.array([1.0, 1.0, 1.0, 1.0])
        self._actuator_trim = np.array([0.0, 0.0, 0.0, 0.0])  # Assuming trim is zero

        self.pwm_max = 1000
        self.pwm_min = 150

        self._control_trim = {
            "ROLL": 0.0,
            "PITCH": 0.0,
            "YAW": 0.0,
            "THRUST_X": 0.0,
            "THRUST_Y": 0.0,
            "THRUST_Z": 0.0,
        }

        self._actuator_sp = np.zeros(self._num_actuators)

    def computeDesaturationGain(self, desaturation_vector, actuator_sp):
        k_min = 0.0
        k_max = 0.0

        for i in range(self._num_actuators):
            if abs(desaturation_vector[i]) < 0.2:
                continue

            if actuator_sp[i] < self._actuator_min[i]:
                k = (self._actuator_min[i] - actuator_sp[i]) / desaturation_vector[i]
                if k < k_min:
                    k_min = k
                if k > k_max:
                    k_max = k

            if actuator_sp[i] > self._actuator_max[i]:
                k = (self._actuator_max[i] - actuator_sp[i]) / desaturation_vector[i]
                if k < k_min:
                    k_min = k
                if k > k_max:
                    k_max = k

        return k_min + k_max

    def desaturateActuators(
        self, actuator_sp, desaturation_vector, increase_only=False
    ):
        gain = self.computeDesaturationGain(desaturation_vector, actuator_sp)

        if increase_only and gain < 0.0:
            return

        for i in range(self._num_actuators):
            actuator_sp[i] += gain * desaturation_vector[i]

        gain = 0.5 * self.computeDesaturationGain(desaturation_vector, actuator_sp)

        for i in range(self._num_actuators):
            actuator_sp[i] += gain * desaturation_vector[i]

    def mixAirmodeDisabled(self, _control_sp, _mix):

        thrust_z = np.zeros(self._num_actuators)
        roll = np.zeros(self._num_actuators)
        pitch = np.zeros(self._num_actuators)

        for i in range(self._num_actuators):
            self._actuator_sp[i] = (
                self._actuator_trim[i]
                + _mix[i, 0] * (_control_sp[0] - self._control_trim["ROLL"])
                + _mix[i, 1] * (_control_sp[1] - self._control_trim["PITCH"])
                + _mix[i, 3] * (_control_sp[3] - self._control_trim["THRUST_X"])
                + _mix[i, 4] * (_control_sp[4] - self._control_trim["THRUST_Y"])
                + _mix[i, 5] * (_control_sp[5] - self._control_trim["THRUST_Z"])
            )
            thrust_z[i] = _mix[i, 5]  # THRUST_Z is at index 5
            roll[i] = _mix[i, 0]  # ROLL is at index 0
            pitch[i] = _mix[i, 1]  # PITCH is at index 1

        self.desaturateActuators(self._actuator_sp, thrust_z, True)
        self.desaturateActuators(self._actuator_sp, roll)
        self.desaturateActuators(self._actuator_sp, pitch)

        return self.mixYaw(_control_sp, _mix)

    def mixYaw(self, _control_sp, _mix):
        yaw = np.zeros(self._num_actuators)
        thrust_z = np.zeros(self._num_actuators)

        for i in range(self._num_actuators):
            self._actuator_sp[i] += _mix[i, 2] * (
                _control_sp[2] - self._control_trim["YAW"]
            )
            yaw[i] = _mix[i, 2]
            thrust_z[i] = _mix[i, 5]

        max_prev = self._actuator_max.copy()
        self._actuator_max += (self._actuator_max - self._actuator_min) * 0.15
        self.desaturateActuators(self._actuator_sp, yaw)
        self._actuator_max = max_prev

        self.desaturateActuators(self._actuator_sp, thrust_z, True)

        pwm_output = []
        for i in range(self._num_actuators):
            pwm_output.append(
                round(
                    self._actuator_sp[i] * (self.pwm_max - self.pwm_min) + self.pwm_min
                )
            )
        return pwm_output
