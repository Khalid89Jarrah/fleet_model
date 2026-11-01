# FLEET MODEL

Closed form rigid body and actuator model for training multirotor rate
controllers across a fleet of airframes.

This repository trains the block of a multirotor rate controller that converts
rate error into a body torque demand, then exports the trained policy as C++
source for a flight stack firmware build. The model is NumPy behind a Gymnasium
interface, with no simulator process and no rendering, and it resamples the
airframe at every episode reset so training covers a distribution of vehicles
rather than one fixed airframe.

Reference implementation for:

> Khaled Jarrah and Osamah A. Rawashdeh. A Learned Rate Controller with In Flight Adaptation, Trained Across Degraded Airframes and Flown with Damaged Rotors on a Low Cost Autopilot.

---

## Pipeline

```
ppo_train.py                        -> ppo_model_<N>_actor.weights.h5
ppo_fm_evaluate.py                  -> evaluation report for a checkpoint
convert_actor_online_tuning_tflite  -> ppo_model.cpp, ppo_model.hpp
```

The emitted `ppo_model.cpp` and `ppo_model.hpp` are compiled into the flight stack
firmware image as a byte array, so the flight controller needs no file system
access at run time.

The exported graph has two outputs. The first is the normalised torque demand
in `[-1, 1]^3`. The second is the activation of the final hidden layer, 64
values, which the firmware side online adaptation consumes from the same
inference call. The converter reads both back through the TensorFlow Lite
interpreter and compares them against a NumPy forward pass before emitting any
C++.

---

## Interfaces

| | |
|---|---|
| Observation | 12 values: measured rate, commanded rate, angular acceleration, clamped error integral |
| Action | 3 values in `[-1, 1]`, normalised body torque demand |
| Control interval | 0.004 s |
| Environment id | `FleetModel-v0` |

Control allocation and sequential desaturation run inside the closed loop, so
the redistribution of control effort under actuator saturation is part of the
training dynamics rather than something applied to the policy output
afterwards.

---

## Repository layout

```
src/sensor_interaction/sensor_interaction/
    __init__.py                           registers the FleetModel-v0 environment
    fleet_model_world.py                  Gymnasium environment, episode plans, reward
    fast_drone_node.py                    rigid body, actuators, sensor chain, randomization
    psude_inverse.py                      control allocation pseudo inverse
    sequentialdesaturationmotoroutput.py  sequential desaturation mixer
src/sensor_interaction/scripts/
    ppo_train.py                          training entry point
    ppo_agent.py                          PPO, actor and critic
    ppo_fm_evaluate.py                    evaluation against the training distribution
src/sensor_interaction/resources/fleet_model_drone/
    f450.yaml                             rotor geometry and coefficients
tflite/
    convert_actor_online_tuning_tflite.py export to TensorFlow Lite and C++
docker/
    Dockerfile, run_docker.sh
```

---

## Environment

```bash
docker build --network=host -f docker/Dockerfile --progress=plain -t fleet_model:latest .
. docker/run_docker.sh IMAGE_ID --src_path ./src --model_path ./model
```

`src` mounts at `/opt/fleet_model_ws/src` and `model` at
`/opt/fleet_model_ws/models`. `PYTHONPATH` is set in the image, so no ROS
workspace build is required.

Outside Docker the requirements are Python 3.12, TensorFlow 2.18, Gymnasium,
NumPy, PyYAML, matplotlib and pandas, with
`PYTHONPATH=<repo>/src/sensor_interaction`.

---

## Training

```bash
cd /opt/fleet_model_ws && python3 src/sensor_interaction/scripts/ppo_train.py
```

Writes to `/opt/fleet_model_ws/models`:

| | |
|---|---|
| `ppo_model_<N>_actor.weights.h5` | actor checkpoint, every 1000 episodes |
| `ppo_model_<N>_critic.weights.h5` | critic checkpoint |
| `ppo_model_<N>_config.npy` | architecture, read back by the converter |
| `rollout_log.csv` | per rollout losses, action and observation statistics |
| `training_log.csv` | per checkpoint evaluation summary |

TensorBoard scalars go to `./ppo_tensorboard/` relative to the working
directory.

```bash
tensorboard --logdir=ppo_tensorboard --port=6006
```

Defaults are 64 parallel environments, a 4096 transition rollout buffer,
minibatch 512, 10 epochs per update, actor learning rate `1e-4` and critic
`5e-4`, both Adam, discount 0.99, GAE 0.95, clip range 0.2. Actor and critic
are separate MLPs of two hidden layers of 64 units with hyperbolic tangent
activation. Only the actor is exported.

Set `FLEET_MODEL_STEP_LOG=1` for a per step CSV. It is large and slows
training.

---

## Evaluation

```bash
python3 src/sensor_interaction/scripts/ppo_fm_evaluate.py \
    --model_path /opt/fleet_model_ws/models/ppo_model_16000
```

Runs hover, step jumps, stick release, slow ramp and a setpoint sweep, and
writes `episode_summary.csv`, four plots and `analysis.txt` to a timestamped
directory under `models/`.

`--seed` controls the setpoint profiles only. The per episode airframe
randomization draws from the global NumPy generator, so results vary between
runs with the same seed.

---

## Export to flight stack 

```bash
python3 tflite/convert_actor_online_tuning_tflite.py \
    --checkpoint 16000 \
    --model_dir /opt/fleet_model_ws/models \
    --out_dir   /opt/fleet_model_ws/models
```

Emits `ppo_actor_<N>_online_tuning_float32.tflite`, `ppo_model.cpp` and
`ppo_model.hpp`. The run aborts if the TensorFlow Lite outputs and the NumPy
reference disagree by more than `--tol`, default `1e-5`, so a model that fails
verification is never written out as C++.

`tflite/` is outside the directory mounted by `run_docker.sh`, so run the
converter on the host, or mount the repository root instead of `src`.

The firmware side, which selects between the learned controller and the
baseline PID and applies the bounded online adaptation, is not part of this
repository.

---

## Demonstration Videos

### 1. PID Crash

https://github.com/user-attachments/assets/f7c7b7f7-2e99-4fc8-b027-11a3e5e40149

### 2. High Altitude Yaw Degradation

https://github.com/user-attachments/assets/ea1448cd-5456-4833-8b28-9c5416c8de44

### 3. Low Altitude Yaw Degradation

https://github.com/user-attachments/assets/75a56f77-6ecf-464e-a7ef-20c03828a6fd

### 4. PPO Hard Maneuvering

https://github.com/user-attachments/assets/9201af05-c6a9-4f05-8e02-c78a9efe203d

---

### Using the model in firmware

1. Copy `ppo_model.cpp` and `ppo_model.hpp` into the rate control module of
   your flight stack and add them to its build.

2. Enable TensorFlow Lite for Microcontrollers in the firmware, and give the
   rate controller enough stack and a small tensor arena to run inference.

3. In the rate controller, load the model and read the torque demand from the
   tau output, using the output index from the header rather than assuming an
   order. Feed it the twelve value observation listed under Interfaces.

4. Keep the existing PID controller and add a parameter to select between the
   two, defaulting to PID so the aircraft falls back if the model does not
   load.

**Notes**
> Both controllers take the same inputs, the measured rate, the commanded rate,
> the angular acceleration and the clamped error integral, and both produce a
> body torque demand in `[-1, 1]` published on the same topic. The learned
> controller replaces only the block that turns rate error into that demand, so
> control allocation, mixing and everything downstream is untouched.

---

## License

MIT, see `LICENSE`.

Two files contain code translated from the PX4 Autopilot project and remain
under the BSD 3-Clause License. See `THIRD_PARTY_LICENSES`.
