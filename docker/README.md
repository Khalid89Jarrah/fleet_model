# fleet_model Docker Environment

A Docker environment for running fleet_model fast model training.

---

## Included Components

This Docker image installs and configures:

- **TensorFlow**
- **Gymnasium**
- **Python virtual environment**

---

## Build the Docker Image

To build the Docker image, run:

```bash
docker build --network=host -f docker/Dockerfile --progress=plain -t fleet_model:latest .
```

## Run the docker images:
```bash
. docker/run_docker.sh IMAGE_ID --src_path fleet_model_src_path --model_path fleet_model_model_path
```

## Run the training
Inside the container:
```bash
cd /opt/fleet_model_ws && python3 src/sensor_interaction/scripts/ppo_train.py
```