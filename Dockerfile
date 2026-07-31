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
RUN if [ -n "${TORCH_INDEX}" ]; then \
      pip3 install --no-cache-dir --index-url "${TORCH_INDEX}" torch torchvision ; \
    else \
      pip3 install --no-cache-dir torch torchvision ; \
    fi \
 && python3 -c "import torch, torchvision; print('OK torch', torch.__version__, 'torchvision', torchvision.__version__)"
