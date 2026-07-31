#!/usr/bin/env bash
# build.sh — 做出「一顆」含 DeepStream + CUDA + PyTorch 的 image, 之後永久重用。
#
# 沒有現成的單一 image 同時有 DeepStream + PyTorch (官方兩者分開出), 所以自己合一顆。
# 但「零編譯」: FROM 現成 deepstream (已含 CUDA/pyds/gstreamer), 只 pip 疊「預建」torch wheel。
# 做一次就好, image 永久存在 docker 裡 (重開機也在)。前置 (裝 docker): 先跑一次 ./setup-host.sh。
#
# 為何不用 `jetson-containers build` (實測, 見 handoff DEP-11/T12): 它一定把依賴樹(含 CUDA)
#   從源碼重建 —— r36.5 沒預建 deepstream 時要編數小時, 且在 cuda 階段 exit 100。這裡直接
#   站在現成 image 上補 torch, CUDA 沿用不重裝。
set -euo pipefail

CONTAINER_NAME="${CONTAINER_NAME:-ds-torch-webcam}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 基底切換 (環境變數):
#   預設 = nvcr.io/nvidia/deepstream:7.1-triton-multiarch
#          NVIDIA 官方對 JP6.2 / L4T 36.4-36.5 的正式 release (cu126, 與 host 原生對齊)。
#          需先登入 NGC (免費帳號 + API key): docker login nvcr.io
#   備援 = dustynv/deepstream:r36.2.0 (jetson-containers r36.2 預建, 免登入;
#          已實測 pyds/gstreamer 在 r36.5 host 全過, 但 CUDA 12.2 較舊):
#            DEEPSTREAM_BASE=dustynv/deepstream:r36.2.0 TORCH_INDEX= ./build.sh
DEEPSTREAM_BASE="${DEEPSTREAM_BASE:-nvcr.io/nvidia/deepstream:7.1-triton-multiarch}"

if ! command -v docker >/dev/null 2>&1; then
  echo "[build] 找不到 docker。請先跑一次: ./setup-host.sh" >&2
  exit 1
fi

echo "[build] 做 image: FROM ${DEEPSTREAM_BASE} + pip torch/torchvision -> ${CONTAINER_NAME}"
docker build -t "${CONTAINER_NAME}" \
  --build-arg "DEEPSTREAM_BASE=${DEEPSTREAM_BASE}" \
  ${TORCH_INDEX+--build-arg "TORCH_INDEX=${TORCH_INDEX}"} \
  -f "${HERE}/Dockerfile" "${HERE}"

echo "[build] 完成。這顆 image 永久存在, 之後直接 ./run.sh preflight / ./run.sh app。"
