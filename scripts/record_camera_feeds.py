"""Record the existing HTTP JPEG feeds without opening the USB cameras."""

import argparse
import json
import signal
import threading
import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np

p = argparse.ArgumentParser()
p.add_argument("output", type=Path)
p.add_argument("--fps", type=float, default=5)
args = p.parse_args()
args.output.mkdir(parents=True, exist_ok=False)
stop = threading.Event()
signal.signal(signal.SIGTERM, lambda *_: stop.set())
signal.signal(signal.SIGINT, lambda *_: stop.set())
start = time.monotonic()
stats = {}


def record(name, number):
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    writer = None
    frames = errors = 0
    deadline = start
    try:
        with (args.output / f"{name}-timestamps.jsonl").open("w", buffering=1) as log:
            while not stop.is_set() and not (args.output / "STOP").exists():
                stop.wait(max(0, deadline - time.monotonic()))
                if stop.is_set():
                    break
                try:
                    with opener.open(f"http://127.0.0.1:8765/camera-{number}.jpg", timeout=2) as r:
                        payload = r.read()
                    frame = cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_COLOR)
                    if frame is None:
                        raise RuntimeError("JPEG decode failed")
                    if writer is None:
                        h, w = frame.shape[:2]
                        writer = cv2.VideoWriter(
                            str(args.output / f"{name}.mp4"),
                            cv2.VideoWriter_fourcc(*"mp4v"),
                            args.fps,
                            (w, h),
                        )
                        if not writer.isOpened():
                            raise RuntimeError("Video writer failed")
                        print(f"RECORDING {name}: {w}x{h} at {args.fps} fps", flush=True)
                    writer.write(frame)
                    frames += 1
                    log.write(
                        json.dumps(
                            {
                                "frame": frames,
                                "monotonic": time.monotonic(),
                                "elapsed": time.monotonic() - start,
                                "utc_epoch": time.time(),
                            }
                        )
                        + "\n"
                    )
                except Exception as exc:
                    errors += 1
                    log.write(
                        json.dumps({"error": str(exc), "elapsed": time.monotonic() - start}) + "\n"
                    )
                deadline += 1 / args.fps
                if deadline < time.monotonic() - 1 / args.fps:
                    deadline = time.monotonic()
    finally:
        if writer is not None:
            writer.release()
        stats[name] = {"frames": frames, "errors": errors}


threads = [
    threading.Thread(target=record, args=(name, num)) for name, num in [("wrist", 1), ("side", 2)]
]
for thread in threads:
    thread.start()
for thread in threads:
    thread.join()
(args.output / "summary.json").write_text(
    json.dumps({"elapsed": time.monotonic() - start, "fps": args.fps, "cameras": stats}, indent=2)
)
print(json.dumps(stats), flush=True)
