#!/usr/bin/env python3
"""
app.py — 最小應用: webcam -> DeepStream/GStreamer -> PyTorch 分類 -> 輸出.

對應 handoff.md 的 AC-2。三個依賴都會被實際用到:
  * GStreamer + DeepStream nv 元件  -> webcam 擷取與 GPU 影像前處理
  * appsink                         -> 把 frame 拉進 numpy
  * PyTorch (torchvision)           -> 對 frame 做 ImageNet 分類

設計原則: 優雅降級。少了某個依賴時, 盡量還能跑到更前面, 並清楚說明少了什麼。
  - 沒有 nvvideoconvert -> 退回一般 videoconvert
  - 沒有 torchvision    -> 不分類, 只驗證擷取路徑 (仍算 pipeline 通)
  - 沒有 DISPLAY        -> headless, 只印文字 (預設)

環境變數:
  VIDEO_DEVICE  (預設 /dev/video0)
  INFER_EVERY   (每幾幀跑一次分類, 預設 15)
  RUN_SECONDS   (跑幾秒後自動結束, 預設 60; 0 = 不限)
  WIDTH,HEIGHT  (擷取解析度, 預設 640x480)
"""

import os
import sys
import time

import numpy as np

# ---- GStreamer ----
try:
    import gi

    gi.require_version("Gst", "1.0")
    from gi.repository import GLib, Gst
except Exception as e:  # noqa: BLE001
    print(f"[FATAL] 無法載入 GStreamer (gi/Gst): {e!r}", file=sys.stderr)
    print("        這是核心依賴。請確認在 DeepStream 容器內執行。", file=sys.stderr)
    sys.exit(2)

VIDEO_DEVICE = os.environ.get("VIDEO_DEVICE", "/dev/video0")
INFER_EVERY = int(os.environ.get("INFER_EVERY", "15"))
RUN_SECONDS = int(os.environ.get("RUN_SECONDS", "60"))
WIDTH = int(os.environ.get("WIDTH", "640"))
HEIGHT = int(os.environ.get("HEIGHT", "480"))


def has_element(name):
    """GStreamer 元件是否註冊 (用來決定走 nv 硬體路徑還是軟體 fallback)。"""
    return Gst.ElementFactory.find(name) is not None


# --------------------------------------------------------------------------
# PyTorch 分類器 (可降級)
# --------------------------------------------------------------------------
class Classifier:
    """MobileNetV3-Small (備援 ResNet18)。torchvision 缺席時自動停用。"""

    def __init__(self):
        self.ready = False
        self.device = "cpu"
        self.labels = None
        try:
            import torch  # noqa: PLC0415
            import torchvision  # noqa: PLC0415
            from torchvision import transforms  # noqa: PLC0415

            self.torch = torch
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

            model, weights = self._load_model(torchvision)
            self.model = model.eval().to(self.device)
            self.labels = weights.meta["categories"]
            self.preprocess = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((224, 224), antialias=True),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                      std=[0.229, 0.224, 0.225]),
            ])
            self.ready = True
            print(f"[classifier] ready on {self.device} "
                  f"({self._name}, {len(self.labels)} classes)")
        except Exception as e:  # noqa: BLE001
            print(f"[classifier] 停用 (torchvision 不可用): {e!r}")
            print("[classifier] app 會繼續跑, 只驗證擷取路徑 (見 handoff DEP-3)。")

    def _load_model(self, tv):
        # 首選 MobileNetV3-Small, 失敗退 ResNet18。weights 首次執行會自動下載。
        try:
            w = tv.models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
            self._name = "mobilenet_v3_small"
            return tv.models.mobilenet_v3_small(weights=w), w
        except Exception:  # noqa: BLE001
            w = tv.models.ResNet18_Weights.IMAGENET1K_V1
            self._name = "resnet18"
            return tv.models.resnet18(weights=w), w

    def classify(self, rgb):
        """rgb: HxWx3 uint8 numpy -> (label, prob)。未就緒時回 None。"""
        if not self.ready:
            return None
        with self.torch.no_grad():
            x = self.preprocess(rgb).unsqueeze(0).to(self.device)
            out = self.model(x)
            prob = self.torch.nn.functional.softmax(out[0], dim=0)
            p, idx = prob.max(0)
            return self.labels[int(idx)], float(p)


# --------------------------------------------------------------------------
# Pipeline 組裝
# --------------------------------------------------------------------------
def build_pipeline():
    """
    v4l2src -> (nvvideoconvert|videoconvert) -> capsfilter(RGBA/RGB) -> appsink

    優先用 DeepStream 的 nvvideoconvert (GPU); 沒有就退回 CPU videoconvert。
    回傳 (pipeline, appsink, path_desc)。
    """
    use_nv = has_element("nvvideoconvert")
    conv = "nvvideoconvert" if use_nv else "videoconvert"
    # nvvideoconvert 輸出走 NVMM->system memory 需再一次 videoconvert 轉 RGB
    if use_nv:
        chain = (
            f"v4l2src device={VIDEO_DEVICE} ! "
            f"video/x-raw,width={WIDTH},height={HEIGHT} ! "
            f"nvvideoconvert ! video/x-raw,format=RGBA ! "
            f"videoconvert ! video/x-raw,format=RGB ! "
            f"appsink name=sink emit-signals=true max-buffers=1 drop=true"
        )
    else:
        chain = (
            f"v4l2src device={VIDEO_DEVICE} ! "
            f"video/x-raw,width={WIDTH},height={HEIGHT} ! "
            f"videoconvert ! video/x-raw,format=RGB ! "
            f"appsink name=sink emit-signals=true max-buffers=1 drop=true"
        )
    path_desc = f"{'nvvideoconvert (GPU/DeepStream)' if use_nv else 'videoconvert (CPU fallback)'}"
    print(f"[pipeline] source={VIDEO_DEVICE} conv={conv}")
    print(f"[pipeline] gst-launch equiv:\n  {chain}")
    pipeline = Gst.parse_launch(chain)
    appsink = pipeline.get_by_name("sink")
    return pipeline, appsink, path_desc


class App:
    def __init__(self):
        Gst.init(None)
        self.clf = Classifier()
        self.pipeline, self.appsink, self.path_desc = build_pipeline()
        self.appsink.connect("new-sample", self.on_sample)
        self.loop = GLib.MainLoop()
        self.frame_count = 0
        self.t0 = time.time()
        self.last_label = "(warming up)"

        bus = self.pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message", self.on_bus)

    def on_bus(self, _bus, msg):
        t = msg.type
        if t == Gst.MessageType.EOS:
            print("[bus] EOS")
            self.stop()
        elif t == Gst.MessageType.ERROR:
            err, dbg = msg.parse_error()
            print(f"[bus][ERROR] {err}: {dbg}", file=sys.stderr)
            print("            常見原因: webcam 未直通 (--device), 或格式不支援。見 handoff DEP-6。",
                  file=sys.stderr)
            self.stop()

    def on_sample(self, sink):
        sample = sink.emit("pull-sample")
        if sample is None:
            return Gst.FlowReturn.ERROR
        buf = sample.get_buffer()
        caps = sample.get_caps().get_structure(0)
        w = caps.get_value("width")
        h = caps.get_value("height")
        ok, minfo = buf.map(Gst.MapFlags.READ)
        if not ok:
            return Gst.FlowReturn.ERROR
        try:
            # RGB, 3 bytes/px。用 copy 避免 unmap 後記憶體失效。
            frame = np.frombuffer(minfo.data, dtype=np.uint8)
            frame = frame[: w * h * 3].reshape(h, w, 3).copy()
        finally:
            buf.unmap(minfo)

        self.frame_count += 1
        if self.clf.ready and self.frame_count % INFER_EVERY == 0:
            res = self.clf.classify(frame)
            if res:
                label, p = res
                self.last_label = f"{label} (p={p:.2f})"
                print(f"[frame {self.frame_count:5d}] {self.last_label}")
        elif not self.clf.ready and self.frame_count % INFER_EVERY == 0:
            print(f"[frame {self.frame_count:5d}] captured {w}x{h} "
                  f"(分類停用, 僅驗證擷取)")

        # 每 ~5 秒印一次 FPS
        if self.frame_count % (INFER_EVERY * 10) == 0:
            fps = self.frame_count / max(time.time() - self.t0, 1e-6)
            print(f"[stats] frames={self.frame_count} fps={fps:.1f} "
                  f"path={self.path_desc}")

        if RUN_SECONDS and (time.time() - self.t0) >= RUN_SECONDS:
            print(f"[app] 已跑 {RUN_SECONDS}s, 收工。")
            self.stop()
        return Gst.FlowReturn.OK

    def stop(self):
        self.pipeline.set_state(Gst.State.NULL)
        if self.loop.is_running():
            self.loop.quit()

    def run(self):
        print(f"[app] 啟動: 影像路徑 = {self.path_desc}")
        print(f"[app] 讀取 {VIDEO_DEVICE}, 每 {INFER_EVERY} 幀分類一次, "
              f"跑 {RUN_SECONDS or '∞'}s。Ctrl-C 中止。")
        self.pipeline.set_state(Gst.State.PLAYING)
        try:
            self.loop.run()
        except KeyboardInterrupt:
            print("\n[app] 收到中斷。")
        finally:
            self.stop()
            elapsed = time.time() - self.t0
            fps = self.frame_count / max(elapsed, 1e-6)
            print(f"[app] 結束。共 {self.frame_count} 幀, {elapsed:.1f}s, 平均 {fps:.1f} FPS。")
            print(f"[app] 最後分類: {self.last_label}")


if __name__ == "__main__":
    App().run()
