#!/bin/bash
IMAGE_ID=$1
shift

while [[ $# -gt 0 ]]; do
  case $1 in
    --src_path) SRC_PATH="$2"; shift 2;;
    --model_path) MODEL_PATH="$2"; shift 2;;
    *) echo "Unknown option: $1"; exit 1;;
  esac
done

# Fail if missing
[[ -z "$IMAGE_ID" || -z "$SRC_PATH" || -z "$MODEL_PATH" ]] && {
  echo "Error: Required args -> IMAGE_ID --src_path PATH --model_path PATH"
return 1 2>/dev/null || exit 1
}

# For GPU add --gpus all to the command below, needs the nvidia container toolkit
sudo docker run -it --privileged \
  --entrypoint=/bin/bash \
  -p 6006:6006/udp \
  --rm \
  -v /dev:/dev \
  -v "$SRC_PATH":/opt/fleet_model_ws/src \
  -v "$MODEL_PATH":/opt/fleet_model_ws/models \
  --net=host \
  --device /dev/dri \
  --env="DISPLAY" \
  --env="QT_X11_NO_MITSHM=1" \
  --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
  $IMAGE_ID
