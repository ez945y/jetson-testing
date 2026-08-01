# 薄層融合: 以現成 deepstream image 為基底, 疊上預建的 torch/torchvision wheel。
#
# 基底 = NVIDIA 官方 DeepStream 7.1 —— 官方對 JetPack 6.2 / L4T 36.4-36.5 的 release (cu126,
# 與 host 原生對齊)。沒有任何現成單一 image 同時含 DeepStream+PyTorch, 所以自己合這一顆;
# 但零編譯: 基底已含 CUDA / gstreamer / deepstream, 只 pip 裝 jetson 預建 torch wheel。
#
# 為何不用 `jetson-containers build --base` (見 handoff DEP-11/T12):
#   那條會把 pytorch 的整條依賴(含 CUDA)在基底上「從源碼重建」, 即使基底已有 CUDA ——
#   實測 10 stages、在 cuda 階段 exit 100 失敗、耗 54 分。
ARG DEEPSTREAM_BASE=nvcr.io/nvidia/deepstream:7.1-triton-multiarch
FROM ${DEEPSTREAM_BASE}

# torch wheel 索引: 官方 DS 基底沒帶 jetson pip 索引, 預設指向 jetson-ai-lab 的 jp6/cu126
# (對齊 DS7.1 的 CUDA 12.6)。若改用 dustynv 基底(內建索引), 可傳空字串。
# 注意: 舊域名 pypi.jetson-ai-lab.dev 已 NXDOMAIN (2026-08 實測), 現行為 .io (見 handoff DEP-12)。
ARG TORCH_INDEX="https://pypi.jetson-ai-lab.io/jp6/cu126"
# 層1: 只下載安裝 wheel (最花時間的一層, 讓它穩定可 cache)
RUN if [ -n "${TORCH_INDEX}" ]; then \
      pip3 install --no-cache-dir --index-url "${TORCH_INDEX}" torch torchvision ; \
    else \
      pip3 install --no-cache-dir torch torchvision ; \
    fi

# 層2: 補 wheel 連結的系統庫 + numpy(app 要用) + 驗證匯入。
# 與層1分開: 之後再發現缺 .so 只重跑這層, 不用重下載幾百MB的 torch (見 handoff DEP-13)。
# 官方 DS 容器只保證 DeepStream 自身依賴, torch wheel 連結的這些都缺 (dustynv 容器則有預裝):
#   - libopenblas0            (apt, Ubuntu 源)
#   - libcudss.so.0 (cuDSS)   (apt 套件 `cudss`, 來自 NVIDIA CUDA 網路源 — 與 jetson-containers
#   - libcusparseLt (cuSPARSELt) (apt 套件 `libcusparselt0`)  install_cudss/cusparselt.sh 同款做法)
RUN apt-get update && apt-get install -y --no-install-recommends libopenblas0 wget ca-certificates \
 && wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/arm64/cuda-keyring_1.1-1_all.deb -O /tmp/cuda-keyring.deb \
 && dpkg -i /tmp/cuda-keyring.deb \
 && apt-get update \
 && apt-get install -y --no-install-recommends cudss libcusparselt0 \
 && dpkg --purge cuda-keyring && rm -f /tmp/cuda-keyring.deb && rm -rf /var/lib/apt/lists/* \
 && ldconfig \
 && pip3 install --no-cache-dir numpy \
 && python3 -c "import torch, torchvision, numpy; print('OK torch', torch.__version__, 'torchvision', torchvision.__version__, 'cuda_built', torch.backends.cuda.is_built())"

# 層3: pyds (DeepStream Python bindings) + python3-gi (app 的 GStreamer python 介面)。
# 官方 DS7.1 容器「不」內建 pyds (實測 preflight FAIL); 官方 wheel 在 deepstream_python_apps releases,
# v1.2.0 = DS 7.1 對應版, cp310 = 容器 Python 3.10。
RUN apt-get update && apt-get install -y --no-install-recommends python3-gi python3-gst-1.0 \
 && rm -rf /var/lib/apt/lists/* \
 && pip3 install --no-cache-dir \
      https://github.com/NVIDIA-AI-IOT/deepstream_python_apps/releases/download/v1.2.0/pyds-1.2.0-cp310-cp310-linux_aarch64.whl \
 && python3 -c "import gi; print('OK gi + pyds installed')"
# 注意: build 階段「不能」import pyds —— 它連結的 libnvbufsurface 等 L4T 庫是 nvidia runtime
# 在 `docker run` 時才從 host 掛進來的, build 時不存在。pyds 的匯入驗證交給 runtime 的 preflight。

# 層4: 把 MobileNetV3 權重「烤進 image」(build 時下載一次)。
# 否則 app 每次啟動都要去 download.pytorch.org 抓 (~10MB), 容器 --rm 不留 cache,
# 且實測 runtime 下載會卡 DNS (見 handoff DEP-15)。烤進去後 runtime 零網路依賴。
RUN python3 -c "import torchvision; \
    torchvision.models.mobilenet_v3_small(weights=torchvision.models.MobileNet_V3_Small_Weights.IMAGENET1K_V1); \
    print('OK weights baked')"

# 層5: 補 gst 音訊外掛缺的庫, 消掉啟動時那排 'Failed to load plugin' 警告 (純觀感, 影像路徑用不到)。
# librivermax (NVIDIA Rivermax, deepstream udp 外掛用) 非 apt 可裝、也用不到, 該條警告保留。
RUN apt-get update && apt-get install -y --no-install-recommends \
      libflac8 libmpg123-0 libmp3lame0 libmjpegutils-2.1-0 libavcodec58 \
 && rm -rf /var/lib/apt/lists/*
