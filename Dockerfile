# 薄層融合: 以現成 deepstream image 為基底, 疊上預建的 torch/torchvision wheel。
#
# 為何不用 `jetson-containers build --base` (見 handoff DEP-11/T12):
#   那條會把 pytorch 的整條依賴(含 CUDA)在基底上「從源碼重建」, 即使基底已有 CUDA ——
#   實測 10 stages、在 cuda 階段 exit 100 失敗、耗 54 分。這裡直接 FROM 現成 deepstream,
#   它已含 CUDA / python / pyds / gstreamer, 只 pip 裝 torch 預建 wheel, 幾分鐘、零編譯。
ARG DEEPSTREAM_BASE=dustynv/deepstream:r36.2.0
FROM ${DEEPSTREAM_BASE}

# dustynv 基底內建 jetson wheel 索引, 直接 pip 就會抓對的 wheel。
# 非 dusty-nv 基底(如 NVIDIA 官方)請用 TORCH_INDEX 指定, 例:
#   https://pypi.jetson-ai-lab.dev/jp6/cu126
ARG TORCH_INDEX=""
RUN if [ -n "${TORCH_INDEX}" ]; then \
      pip3 install --no-cache-dir --index-url "${TORCH_INDEX}" torch torchvision ; \
    else \
      pip3 install --no-cache-dir torch torchvision ; \
    fi \
 && python3 -c "import torch, torchvision; print('OK torch', torch.__version__, 'torchvision', torchvision.__version__)"
