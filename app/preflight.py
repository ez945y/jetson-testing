#!/usr/bin/env python3
"""
preflight.py — 依賴健檢 / dependency completeness check.

這支就是「套件依賴完不完整」的驗收工具（對應 handoff.md 的 AC-1）。
設計原則：**永遠跑到底、永遠不 crash**。每一項檢查獨立 try/except，
最後印出一張清楚的記分板，讓人一眼看出缺哪個依賴。

Exit code 0 = 全過；非 0 = 有必要項失敗（可接進 CI）。
"""

import os
import shutil
import subprocess
import sys

# ANSI 顏色（非 TTY 時自動關閉，避免 log 裡一堆亂碼）
_C = sys.stdout.isatty()
GREEN = "\033[32m" if _C else ""
RED = "\033[31m" if _C else ""
YEL = "\033[33m" if _C else ""
DIM = "\033[2m" if _C else ""
RST = "\033[0m" if _C else ""

results = []  # (name, ok: bool|None, detail, required: bool)


def record(name, ok, detail="", required=True):
    results.append((name, ok, detail, required))
    tag = f"{GREEN}PASS{RST}" if ok else (f"{RED}FAIL{RST}" if ok is False else f"{YEL}WARN{RST}")
    print(f"  [{tag}] {name:<28} {DIM}{detail}{RST}")


def run(cmd):
    """執行外部指令，回傳 (returncode, stdout+stderr)。找不到指令不丟例外。"""
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        return p.returncode, (p.stdout + p.stderr).strip()
    except FileNotFoundError:
        return 127, f"command not found: {cmd[0]}"
    except Exception as e:  # noqa: BLE001
        return 1, str(e)


def section(title):
    print(f"\n{title}")


def main():
    print("=" * 62)
    print(" Jetson DeepStream + PyTorch + Webcam — 依賴健檢 (preflight)")
    print("=" * 62)

    # ---- 環境基本資訊 ----
    section("環境 / environment")
    record("python", True, f"{sys.version.split()[0]}", required=False)
    l4t = ""
    if os.path.exists("/etc/nv_tegra_release"):
        try:
            with open("/etc/nv_tegra_release") as f:
                l4t = f.readline().strip()
        except OSError:
            pass
    record("L4T (nv_tegra_release)", bool(l4t) or None,
           l4t or "not a Jetson? file missing", required=False)

    # ---- AC-1 #1,#2: PyTorch + CUDA ----
    section("PyTorch")
    try:
        import torch  # noqa: PLC0415
        record("import torch", True, f"torch {torch.__version__}")
        cuda_ok = torch.cuda.is_available()
        dev = ""
        if cuda_ok:
            try:
                dev = torch.cuda.get_device_name(0)
            except Exception as e:  # noqa: BLE001
                dev = f"(get_device_name failed: {e})"
        record("torch.cuda.is_available", cuda_ok, dev or "no CUDA device seen")
    except Exception as e:  # noqa: BLE001
        record("import torch", False, repr(e))

    try:
        import torchvision  # noqa: PLC0415
        record("import torchvision", True, f"torchvision {torchvision.__version__}",
               required=False)
    except Exception as e:  # noqa: BLE001
        # torchvision 常因與 torch ABI 不合而炸 —— 標為 WARN，app 會降級
        record("import torchvision", None, f"降級可用: {e!r}", required=False)

    # ---- AC-1 #3: DeepStream Python bindings ----
    section("DeepStream / pyds")
    try:
        import pyds  # noqa: PLC0415
        ver = getattr(pyds, "__version__", "unknown")
        record("import pyds", True, f"pyds {ver}")
    except Exception as e:  # noqa: BLE001
        record("import pyds", False, repr(e))

    rc, out = run(["deepstream-app", "--version"])
    line = out.splitlines()[0] if out else ""
    record("deepstream-app --version", rc == 0, line or out[:60])

    # ---- AC-1 #4,#5: GStreamer 與 nv 外掛 ----
    section("GStreamer 外掛")
    if shutil.which("gst-inspect-1.0") is None:
        record("gst-inspect-1.0", False, "not found (GStreamer 未安裝?)")
    else:
        for plugin in ("v4l2src", "nvvideoconvert", "nvstreammux", "nvinfer", "appsink"):
            rc, _out = run(["gst-inspect-1.0", plugin])
            # nvinfer / nvstreammux 屬 DeepStream，非必要但強烈期望；核心的 v4l2src/appsink 必要
            req = plugin in ("v4l2src", "appsink")
            record(f"gst plugin: {plugin}", rc == 0,
                   "available" if rc == 0 else "missing", required=req)

    # ---- AC-1 #6: Webcam 裝置 ----
    section("Webcam")
    dev = os.environ.get("VIDEO_DEVICE", "/dev/video0")
    exists = os.path.exists(dev)
    openable = False
    detail = "not present"
    if exists:
        detail = "present"
        try:
            fd = os.open(dev, os.O_RDONLY | os.O_NONBLOCK)
            os.close(fd)
            openable = True
            detail = "present & openable"
        except OSError as e:
            detail = f"present but not openable: {e}"
    record(f"video device {dev}", exists and openable, detail)

    # ---- 記分板 ----
    print("\n" + "=" * 62)
    required_fail = [n for n, ok, _d, req in results if req and ok is False]
    warns = [n for n, ok, _d, _req in results if ok is None]
    passed = sum(1 for _n, ok, _d, _req in results if ok is True)
    total = len(results)
    print(f" 結果: {passed}/{total} 通過, {len(warns)} 警告, {len(required_fail)} 必要項失敗")
    if required_fail:
        print(f"{RED} 必要依賴缺失: {', '.join(required_fail)}{RST}")
        print(" → 依賴不完整。請對照 handoff.md §3 的 DEP 表處理。")
    else:
        print(f"{GREEN} 所有必要依賴齊備，可以跑 app.py。{RST}")
    print("=" * 62)

    return 1 if required_fail else 0


if __name__ == "__main__":
    sys.exit(main())
