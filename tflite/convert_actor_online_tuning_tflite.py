import argparse
import os
import re
import sys
import h5py
import numpy as np
import tensorflow as tf

DEFAULT_ARCH = {"obs_dim": 9, "act_dim": 3, "hidden": [64, 64]}

parser = argparse.ArgumentParser(
    description="Convert a PPO actor checkpoint to a two output TFLite blob "
    "(tau, hidden tap) and emit ppo_model.cpp / ppo_model.hpp for PX4"
)
parser.add_argument("--checkpoint", type=int, required=True)
parser.add_argument("--model_dir", type=str, default=".")
parser.add_argument("--out_dir", type=str, default=".")
parser.add_argument("--tol", type=float, default=1e-5)
parser.add_argument("--n_verify", type=int, default=1024)
args = parser.parse_args()

ckpt = args.checkpoint
model_dir = args.model_dir
out_dir = args.out_dir
os.makedirs(out_dir, exist_ok=True)

arch = DEFAULT_ARCH.copy()
config_path = os.path.join(model_dir, f"ppo_model_{ckpt}_config.npy")
if os.path.exists(config_path):
    cfg = np.load(config_path, allow_pickle=True).item()
    for k in ("obs_dim", "act_dim", "hidden"):
        if k in cfg:
            arch[k] = cfg[k]
    print(f"[arch] loaded config from {config_path}")
else:
    print(f"[arch] no config, using defaults: {arch}")

OBS_DIM = arch["obs_dim"]
ACT_DIM = arch["act_dim"]
HIDDEN = arch["hidden"]
HIDDEN_DIM = HIDDEN[-1]
print(
    f"[arch] obs={OBS_DIM}  act={ACT_DIM}  hidden={HIDDEN}  hidden_tap_dim={HIDDEN_DIM}"
)

weights_path = os.path.join(model_dir, f"ppo_model_{ckpt}_actor.weights.h5")
if not os.path.exists(weights_path):
    sys.exit(f"[error] weights file not found: {weights_path}")

raw = {}
with h5py.File(weights_path, "r") as f:
    for i in range(len(HIDDEN)):
        raw[f"h{i}_W"] = np.asarray(f[f"h{i}/vars/0"])
        raw[f"h{i}_b"] = np.asarray(f[f"h{i}/vars/1"])
    raw["mean_W"] = np.asarray(f["layers/dense_2/vars/0"])
    raw["mean_b"] = np.asarray(f["layers/dense_2/vars/1"])
    if "vars/0" in f:
        raw["log_std"] = np.asarray(f["vars/0"])
    else:
        raw["log_std"] = np.zeros((1, ACT_DIM), dtype=np.float32)

expected = []
prev = OBS_DIM
for i, units in enumerate(HIDDEN):
    expected.append((f"h{i}_W", (prev, units)))
    expected.append((f"h{i}_b", (units,)))
    prev = units
expected.append(("mean_W", (prev, ACT_DIM)))
expected.append(("mean_b", (ACT_DIM,)))
for name, exp_shape in expected:
    if tuple(raw[name].shape) != tuple(exp_shape):
        sys.exit(
            f"[error] {name} shape mismatch: expected {exp_shape}, got {raw[name].shape}"
        )
print(f"[weights] all shapes verified")


def build_deploy_model():
    inp = tf.keras.Input(shape=(OBS_DIM,), name="obs")
    x = inp
    hidden_taps = []
    for i, units in enumerate(HIDDEN):
        x = tf.keras.layers.Dense(units, activation="tanh", name=f"h{i}")(x)
        hidden_taps.append(x)
    last_hidden = hidden_taps[-1]
    mean = tf.keras.layers.Dense(ACT_DIM, name="mean_head")(last_hidden)
    tau = tf.keras.layers.Lambda(lambda t: tf.clip_by_value(t, -1.0, 1.0), name="tau")(
        mean
    )
    return tf.keras.Model(
        inputs=inp, outputs=[tau, last_hidden], name="ppo_actor_online_tuning_deploy"
    )


km = build_deploy_model()
for i in range(len(HIDDEN)):
    km.get_layer(f"h{i}").set_weights([raw[f"h{i}_W"], raw[f"h{i}_b"]])
km.get_layer("mean_head").set_weights([raw["mean_W"], raw["mean_b"]])
print("[keras] weights injected by name")


def numpy_forward(obs_batch):
    x = obs_batch.astype(np.float32)
    for i in range(len(HIDDEN)):
        x = np.tanh(x @ raw[f"h{i}_W"] + raw[f"h{i}_b"])
    hidden = x.copy()
    mean = x @ raw["mean_W"] + raw["mean_b"]
    tau = np.clip(mean, -1.0, 1.0)
    return tau, hidden


converter = tf.lite.TFLiteConverter.from_keras_model(km)
tflite_blob = converter.convert()

tflite_path = os.path.join(out_dir, f"ppo_actor_{ckpt}_online_tuning_float32.tflite")
with open(tflite_path, "wb") as f:
    f.write(tflite_blob)
print(f"[tflite] saved {tflite_path} ({len(tflite_blob)} bytes)")

np.random.seed(0)
test_obs = np.random.uniform(-3.0, 3.0, size=(args.n_verify, OBS_DIM)).astype(
    np.float32
)
test_obs[0] = np.zeros(OBS_DIM, dtype=np.float32)
test_obs[1] = np.ones(OBS_DIM, dtype=np.float32)
test_obs[2] = -np.ones(OBS_DIM, dtype=np.float32)

y_np_tau, y_np_h = numpy_forward(test_obs)
y_keras_out = km(test_obs)
y_keras_tau = y_keras_out[0].numpy()
y_keras_h = y_keras_out[1].numpy()

ip = tf.lite.Interpreter(model_content=tflite_blob)
ip.allocate_tensors()
inp_details = ip.get_input_details()[0]
out_details = ip.get_output_details()
print(f"[tflite] {len(out_details)} outputs detected:")
for od in out_details:
    print(
        f"          name={od['name']}  shape={od['shape']}  dtype={od['dtype'].__name__}"
    )

if len(out_details) != 2:
    sys.exit(f"[error] expected 2 outputs (tau, hidden), got {len(out_details)}")

# Matched by shape because TFLite reorders outputs by name
tau_idx = next(i for i, od in enumerate(out_details) if od["shape"][-1] == ACT_DIM)
h_idx = next(i for i, od in enumerate(out_details) if od["shape"][-1] == HIDDEN_DIM)
print(f"[tflite] tau output index={tau_idx}, hidden output index={h_idx}")

y_tfl_tau = np.zeros_like(y_np_tau)
y_tfl_h = np.zeros_like(y_np_h)
for i, o in enumerate(test_obs):
    ip.set_tensor(inp_details["index"], o.reshape(1, OBS_DIM).astype(np.float32))
    ip.invoke()
    y_tfl_tau[i] = ip.get_tensor(out_details[tau_idx]["index"])[0]
    y_tfl_h[i] = ip.get_tensor(out_details[h_idx]["index"])[0]

dnk_tau = np.abs(y_np_tau - y_keras_tau).max()
dnt_tau = np.abs(y_np_tau - y_tfl_tau).max()
dnk_h = np.abs(y_np_h - y_keras_h).max()
dnt_h = np.abs(y_np_h - y_tfl_h).max()

print(f"\n=== Equivalence on {args.n_verify} random obs ===")
print(f"  tau    : max|np-keras|={dnk_tau:.2e}   max|np-tflite|={dnt_tau:.2e}")
print(f"  hidden : max|np-keras|={dnk_h:.2e}     max|np-tflite|={dnt_h:.2e}")
print(f"  tolerance = {args.tol:.1e}")

if max(dnk_tau, dnt_tau, dnk_h, dnt_h) > args.tol:
    sys.exit("[error] tolerance exceeded, model not safe to deploy")
print("[verify] PASSED")

zero_tau, zero_h = numpy_forward(np.zeros((1, OBS_DIM), dtype=np.float32))
print(f"\n=== Idle check (obs=0) ===")
print(f"  tau:    [{zero_tau[0,0]:+.4f}, {zero_tau[0,1]:+.4f}, {zero_tau[0,2]:+.4f}]")
print(
    f"  hidden: rms={np.sqrt(np.mean(zero_h**2)):.3f}  "
    f"min={zero_h.min():+.3f}  max={zero_h.max():+.3f}"
)


def write_cpp_blob(blob, cpp_path, tau_idx, h_idx):
    n = len(blob)
    with open(cpp_path, "w") as f:
        f.write(
            "// Auto generated by convert_actor_online_tuning_tflite.py, DO NOT EDIT.\n"
        )
        f.write('#include "ppo_model.hpp"\n\n')
        f.write("alignas(8) const unsigned char ppo_model_tflite[] = {\n")
        for i in range(0, n, 12):
            chunk = blob[i : i + 12]
            line = "    " + ", ".join(f"0x{b:02x}" for b in chunk)
            if i + 12 < n:
                line += ","
            f.write(line + "\n")
        f.write("};\n")
        f.write(f"\nconst unsigned int ppo_model_tflite_len = {n};\n")
        f.write(f"\nconst int ppo_model_tau_output_index    = {tau_idx};\n")
        f.write(f"const int ppo_model_hidden_output_index = {h_idx};\n")


def write_cpp_header(hpp_path):
    with open(hpp_path, "w") as f:
        f.write("#pragma once\n")
        f.write("// output[ppo_model_tau_output_index]    : 3 floats, tau in [-1,+1]\n")
        f.write(
            f"// output[ppo_model_hidden_output_index] : {HIDDEN_DIM} floats, last hidden tanh activation\n"
        )
        f.write("extern const unsigned char ppo_model_tflite[];\n")
        f.write("extern const unsigned int  ppo_model_tflite_len;\n")
        f.write("extern const int           ppo_model_tau_output_index;\n")
        f.write("extern const int           ppo_model_hidden_output_index;\n")


cpp_path = os.path.join(out_dir, "ppo_model.cpp")
hpp_path = os.path.join(out_dir, "ppo_model.hpp")
write_cpp_blob(tflite_blob, cpp_path, tau_idx, h_idx)
write_cpp_header(hpp_path)
print(f"\n[emit] {cpp_path}")
print(f"[emit] {hpp_path}")

with open(cpp_path) as f:
    text = f.read()
arr_section = text.split("ppo_model_tflite[] = {")[1].split("};")[0]
roundtrip = bytes(int(h, 16) for h in re.findall(r"0x([0-9a-fA-F]{2})", arr_section))
assert roundtrip == tflite_blob, "round trip byte mismatch"
print(f"[verify] cpp matches tflite blob: {len(roundtrip)} bytes")

print("\nDone, drop into PX4:")
print(f"    cp {cpp_path} /opt/PX4-Autopilot/src/modules/mc_rate_control/")
print(f"    cp {hpp_path} /opt/PX4-Autopilot/src/modules/mc_rate_control/")
