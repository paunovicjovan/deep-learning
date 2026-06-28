"""
WebSocket backend that runs sign detection on the app's camera frames.

The app connects to /ws/detect, sends a config message with the frame format
(jpeg or raw BGR), then streams frames one at a time. For each frame the server
replies with the new signs found in it, and the app waits for that reply before
sending the next frame.

Run with:
    python backend/main.py
    uvicorn backend.main:app --host 0.0.0.0 --port 8000
"""

import asyncio
import base64
import sys
from contextlib import asynccontextmanager
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

# Make the repo root importable so `from tracker import ...` works.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from tracker import SignTracker  # noqa: E402

MODEL_PATH = REPO_ROOT / "models" / "best.pt"
TRACKER_CONFIG = REPO_ROOT / "tracker" / "deepsort.yaml"
CONF_THRESHOLD = 0.45
# Downscale frames to this width before inference (YOLO was trained at 640).
TARGET_WIDTH = 640

# One shared tracker, loaded once at startup. We serve one client at a time.
_tracker: SignTracker | None = None
_busy = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _tracker
    print(f"[backend] Loading model: {MODEL_PATH}")
    _tracker = SignTracker(
        model_path=str(MODEL_PATH),
        conf_threshold=CONF_THRESHOLD,
        tracker_config=str(TRACKER_CONFIG),
    )
    print("[backend] Model loaded. Ready for connections on /ws/detect")
    yield
    print("[backend] Shutting down.")


app = FastAPI(title="Traffic Sign Detector backend", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok", "busy": _busy}


def _notification_to_dict(n) -> dict:
    # Convert numpy scalars to native Python types so JSON can serialize them.
    return {
        "track_id": int(n.track_id),
        "class_name": str(n.class_name),
        "description": str(n.description),
        "confidence": float(n.confidence),
        "bbox": [float(v) for v in n.bbox],  # normalized (x1, y1, x2, y2)
        "frame_number": int(n.frame_number),
        "timestamp": float(n.timestamp),
        "image_b64": n.image_b64,  # base64 JPEG crop of the sign
        # True if this is a clearer crop for an already-reported sign.
        "is_update": bool(getattr(n, "is_update", False)),
    }


@app.websocket("/ws/detect")
async def detect(ws: WebSocket):
    global _busy
    await ws.accept()

    if _busy:
        await ws.send_json({"type": "error", "message": "Server is busy with another client."})
        await ws.close()
        return
    if _tracker is None:
        await ws.send_json({"type": "error", "message": "Model not loaded yet."})
        await ws.close()
        return

    _busy = True
    loop = asyncio.get_running_loop()
    try:
        # Handshake: read the frame format.
        cfg = await ws.receive_json()
        fmt = str(cfg.get("format", "raw")).lower()
        if fmt == "raw":
            width = int(cfg["width"])
            height = int(cfg["height"])
            channels = int(cfg.get("channels", 3))
            expected = width * height * channels
        elif fmt != "jpeg":
            await ws.send_json({"type": "error", "message": f"Unknown format: {fmt}"})
            await ws.close()
            return

        _tracker.reset()
        await ws.send_json({"type": "ready", "format": fmt})
        print(f"[backend] Client connected. Format: {fmt}")

        # Stream loop. JPEG frames may arrive as binary or base64 text.
        while True:
            message = await ws.receive()
            if message["type"] == "websocket.disconnect":
                raise WebSocketDisconnect(message.get("code") or 1000)

            data = message.get("bytes")
            if data is None and message.get("text") is not None:
                try:
                    data = base64.b64decode(message["text"])
                except Exception:
                    data = None

            warning = None
            frame = None
            if not data:
                warning = "empty or undecodable message"
            elif fmt == "jpeg":
                frame = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
                if frame is None:
                    warning = "JPEG decode failed"
                elif frame.shape[1] > TARGET_WIDTH:
                    new_h = max(1, round(frame.shape[0] * TARGET_WIDTH / frame.shape[1]))
                    frame = cv2.resize(frame, (TARGET_WIDTH, new_h), interpolation=cv2.INTER_AREA)
            else:
                if len(data) != expected:
                    warning = f"expected {expected} bytes, got {len(data)}"
                else:
                    # frombuffer is read-only; copy so OpenCV/torch can write to it.
                    frame = np.frombuffer(data, dtype=np.uint8).reshape((height, width, channels)).copy()

            if frame is None:
                print(f"[backend] BAD FRAME: {warning} "
                      f"(message had {len(data) if data else 0} bytes)")
                # Reply anyway so the client keeps sending frames.
                await ws.send_json(
                    {"type": "result", "frame": _tracker._frame_count, "detections": [],
                     "warning": warning}
                )
                continue

            frame_h, frame_w = frame.shape[:2]

            # YOLO + DeepSORT is blocking; run it off the event loop.
            notifications = await loop.run_in_executor(
                None, _tracker.process_frame, frame, frame_w, frame_h
            )

            print(f"[backend] frame={_tracker._frame_count} "
                  f"recv={len(data)}B size={frame_w}x{frame_h} "
                  f"new_signs={len(notifications)}")
            for n in notifications:
                print(f"[backend]   -> NEW SIGN ID={n.track_id} {n.class_name} "
                      f"conf={n.confidence:.2f} | {n.description}")

            await ws.send_json({
                "type": "result",
                "frame": _tracker._frame_count,
                "detections": [_notification_to_dict(n) for n in notifications],
            })

    except WebSocketDisconnect:
        print("[backend] Client disconnected.")
    except Exception as e:  # noqa: BLE001
        print(f"[backend] Error: {e}")
        try:
            await ws.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass
    finally:
        _busy = False


if __name__ == "__main__":
    import uvicorn

    # Pass the import string so `backend.main` resolves from any directory.
    sys.path.insert(0, str(REPO_ROOT))
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000)
