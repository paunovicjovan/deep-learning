"""
SignTracker runs YOLO detection + DeepSORT tracking and emits one notification
per new sign.

Each sign is notified only once, when its track is confirmed. To avoid duplicate
alerts when DeepSORT loses and re-acquires the same sign under a new id, a
notification is suppressed if another sign of the same class was seen within the
last class_cooldown_seconds.
"""

from dataclasses import dataclass, field
from typing import Callable, Optional
import time


CLASS_NAMES = [
    "pl80", "p6", "ph", "w", "pa", "p27", "i5", "p1", "il70", "p5",
    "pm", "p19", "ip", "p11", "p13", "p26", "i2", "pn", "p10", "p23",
    "pbp", "p3", "p12", "pne", "i4", "pb", "pg", "pr", "pl5", "pl10",
    "pl15", "pl20", "pl25", "pl30", "pl35", "pl40", "pl50", "pl60",
    "pl65", "pl70", "pl90", "pl100", "pl110", "pl120", "il50", "il60",
    "il80", "il90", "il100", "il110",
]

SIGN_DESCRIPTIONS = {
    "pl5": "Speed limit: 5 km/h",
    "pl10": "Speed limit: 10 km/h",
    "pl15": "Speed limit: 15 km/h",
    "pl20": "Speed limit: 20 km/h",
    "pl25": "Speed limit: 25 km/h",
    "pl30": "Speed limit: 30 km/h",
    "pl35": "Speed limit: 35 km/h",
    "pl40": "Speed limit: 40 km/h",
    "pl50": "Speed limit: 50 km/h",
    "pl60": "Speed limit: 60 km/h",
    "pl65": "Speed limit: 65 km/h",
    "pl70": "Speed limit: 70 km/h",
    "pl80": "Speed limit: 80 km/h",
    "pl90": "Speed limit: 90 km/h",
    "pl100": "Speed limit: 100 km/h",
    "pl110": "Speed limit: 110 km/h",
    "pl120": "Speed limit: 120 km/h",
    "il50": "Minimum speed: 50 km/h",
    "il60": "Minimum speed: 60 km/h",
    "il70": "Minimum speed: 70 km/h",
    "il80": "Minimum speed: 80 km/h",
    "il90": "Minimum speed: 90 km/h",
    "il100": "Minimum speed: 100 km/h",
    "il110": "Minimum speed: 110 km/h",
    "p1": "No Overtaking",
    "p3": "No buses",
    "p5": "No U-turn",
    "p6": "No bicycles",
    "p10": "No motor vehicles",
    "p11": "No horn",
    "p12": "No motorcycles",
    "p13": "No motor vehicles and tractors",
    "p19": "No right turn",
    "p23": "No left turn",
    "p26": "No trucks",
    "p27": "No vehicles carrying explosives",
    "pa": "Axle weight limit",
    "pb": "No Entry for Vehicles",
    "pbp": "No bicycles and pedestrians",
    "ph": "Height limit",
    "pm": "Weight limit",
    "pn": "No stopping and parking",
    "pne": "No entry",
    "pr": "End of speed limit",
    "pg": "Yield / Give way",
    "i2": "Bicycle lane",
    "i4": "Motor vehicles only",
    "i5": "Go straight or turn right",
    "ip": "Pedestrian crossing",
    "w": "Warning",
}

DISPLAY_TTL = 8


@dataclass
class TrackInfo:
    track_id: int
    class_name: str
    confidence: float
    first_seen_frame: int
    last_seen_frame: int
    notified: bool = False
    bbox: tuple = field(default_factory=tuple)  # (x1, y1, x2, y2)


@dataclass
class SignNotification:
    track_id: int
    class_name: str
    description: str
    confidence: float
    bbox: tuple          # normalized 0-1
    timestamp: float
    frame_number: int
    # base64 JPEG of the sign cropped from the frame. None if cropping failed.
    image_b64: Optional[str] = None
    # True if this is a clearer-crop update for an already-notified sign rather
    # than a new detection. Shares track_id + timestamp with the original.
    is_update: bool = False


@dataclass
class EmittedSign:
    """A notified sign we keep around to capture a better crop as it gets closer,
    then emit as an image update once it leaves the frame. Keyed by class_name so
    a re-acquired track (new id) still updates the original notification."""
    track_id: int
    timestamp: float
    class_name: str
    description: str
    confidence: float
    best_area: float
    best_image_b64: Optional[str]
    last_activity_frame: int
    improved: bool = False  # True once a larger crop replaced the initial one


class SignTracker:

    def __init__(
        self,
        model_path: str,
        conf_threshold: float = 0.45,
        iou_threshold: float = 0.45,
        tracker_config: str = "tracker/deepsort.yaml",
        class_cooldown_seconds: Optional[float] = None,
        min_box_size: Optional[float] = None,
        notify_callback: Optional[Callable[[SignNotification], None]] = None,
        verbose: bool = False,
    ):
        from ultralytics import YOLO
        import yaml
        from deep_sort_realtime.deepsort_tracker import DeepSort

        self.model = YOLO(model_path)
        self.conf = conf_threshold
        self.iou = iou_threshold
        self.notify_callback = notify_callback
        self.verbose = verbose

        with open(tracker_config, "r") as f:
            cfg = yaml.safe_load(f)

        # Keep the kwargs so reset() can rebuild a clean tracker per session.
        self._DeepSort = DeepSort
        self._deepsort_kwargs = dict(
            max_age=cfg.get("max_age", 60),
            n_init=cfg.get("n_init", 3),
            max_cosine_distance=cfg.get("max_cosine_distance", 0.4),
            max_iou_distance=cfg.get("max_iou_distance", 0.7),
            nn_budget=cfg.get("nn_budget", None),
            embedder=cfg.get("embedder", "mobilenet"),
            embedder_gpu=cfg.get("embedder_gpu", False),
        )
        self.deepsort = DeepSort(**self._deepsort_kwargs)

        # Explicit arg wins; otherwise fall back to the YAML, then a sane default.
        self.class_cooldown_seconds = (
            class_cooldown_seconds
            if class_cooldown_seconds is not None
            else cfg.get("class_cooldown_seconds", 3.0)
        )

        # Drop boxes whose shorter side is under this many px (far/pixelated
        # signs). 0 disables the filter.
        self.min_box_size = (
            min_box_size
            if min_box_size is not None
            else cfg.get("min_box_size", 0)
        )

        self._active_tracks: dict[int, TrackInfo] = {}

        # prevents duplicate notifications for the same track_id
        self._notified_ids: set[int] = set()

        # class_name -> last time it was seen, used for the class cooldown.
        self._last_class_activity: dict[str, float] = {}

        # class_name -> EmittedSign for notified signs, used to push a better crop.
        self._emitted_signs: dict[str, EmittedSign] = {}

        self._frame_count = 0

    def _get_description(self, class_name: str) -> str:
        return SIGN_DESCRIPTIONS.get(class_name, f"Traffic sign: {class_name}")

    def _crop_sign(self, frame, bbox, pad: float = 0.2) -> Optional[str]:
        """Crop the sign (with some padding) and return it as base64 JPEG, or
        None if the crop is empty."""
        import base64
        import cv2

        h, w = frame.shape[:2]
        x1, y1, x2, y2 = bbox
        bw, bh = x2 - x1, y2 - y1
        if bw <= 0 or bh <= 0:
            return None

        cx1 = max(0, int(x1 - bw * pad))
        cy1 = max(0, int(y1 - bh * pad))
        cx2 = min(w, int(x2 + bw * pad))
        cy2 = min(h, int(y2 + bh * pad))
        if cx2 <= cx1 or cy2 <= cy1:
            return None

        crop = frame[cy1:cy2, cx1:cx2]
        if crop.size == 0:
            return None

        ok, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ok:
            return None
        return base64.b64encode(buf.tobytes()).decode("ascii")

    def _make_notification(self, track: TrackInfo, frame, frame_w: int, frame_h: int) -> SignNotification:
        x1, y1, x2, y2 = track.bbox
        return SignNotification(
            track_id=track.track_id,
            class_name=track.class_name,
            description=self._get_description(track.class_name),
            confidence=round(track.confidence, 3),
            bbox=(
                round(x1 / frame_w, 4),
                round(y1 / frame_h, 4),
                round(x2 / frame_w, 4),
                round(y2 / frame_h, 4),
            ),
            timestamp=time.time(),
            frame_number=self._frame_count,
            image_b64=self._crop_sign(frame, track.bbox),
        )

    def process_frame(self, frame, frame_w: int = None, frame_h: int = None) -> list[SignNotification]:
        self._frame_count += 1
        now = time.time()
        h, w = frame.shape[:2]
        frame_w = frame_w or w
        frame_h = frame_h or h

        results = self.model.predict(
            frame,
            conf=self.conf,
            iou=self.iou,
            verbose=False,
        )

        detections = []
        if results and results[0].boxes is not None:
            for box in results[0].boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf.item())
                class_id = int(box.cls.item())
                class_name = CLASS_NAMES[class_id] if class_id < len(CLASS_NAMES) else f"class_{class_id}"

                # Skip boxes too small to classify reliably yet; picked up later
                # once the sign is closer.
                box_side = min(x2 - x1, y2 - y1)
                if self.min_box_size and box_side < self.min_box_size:
                    if self.verbose:
                        print(
                            f"[Frame {self._frame_count}] too small, skipped | "
                            f"{class_name} | side={box_side:.0f}px "
                            f"< {self.min_box_size:.0f}px | conf={conf:.2f}"
                        )
                    continue

                detections.append(([x1, y1, x2 - x1, y2 - y1], conf, class_name))

        tracks = self.deepsort.update_tracks(detections, frame=frame)

        notifications = []

        for track in tracks:
            if not track.is_confirmed():
                continue

            # Skip coasting tracks (kept alive by Kalman prediction with no real
            # detection this frame) so boxes stay anchored to actual detections.
            if track.time_since_update > 0:
                continue

            track_id = track.track_id
            # orig=True gives the raw detection bbox, not the Kalman-smoothed one.
            x1, y1, x2, y2 = track.to_ltrb(orig=True)
            class_name = track.get_det_class() or "unknown"
            confidence = track.get_det_conf() or 0.0
            if confidence is None:
                confidence = 0.0

            # Box area as a proxy for how close/clear the sign is.
            area = max(0.0, (x2 - x1)) * max(0.0, (y2 - y1))

            if track_id in self._active_tracks:
                t = self._active_tracks[track_id]
                t.last_seen_frame = self._frame_count
                t.confidence = max(t.confidence, confidence)
                t.bbox = (x1, y1, x2, y2)
            else:
                t = TrackInfo(
                    track_id=track_id,
                    class_name=class_name,
                    confidence=confidence,
                    first_seen_frame=self._frame_count,
                    last_seen_frame=self._frame_count,
                    bbox=(x1, y1, x2, y2),
                )
                self._active_tracks[track_id] = t

            if track_id not in self._notified_ids:
                self._notified_ids.add(track_id)
                t.notified = True

                # Suppress if a same-class sign was active recently (likely the
                # same sign re-acquired under a new track_id).
                recently_active = (
                    class_name in self._last_class_activity
                    and now - self._last_class_activity[class_name] < self.class_cooldown_seconds
                )

                if recently_active:
                    if self.verbose:
                        print(
                            f"[Frame {self._frame_count}] DUPLICATE suppressed | "
                            f"ID={track_id} | {class_name}"
                        )
                else:
                    notification = self._make_notification(t, frame, frame_w, frame_h)
                    notifications.append(notification)

                    # Start tracking the best crop for this sign.
                    self._emitted_signs[class_name] = EmittedSign(
                        track_id=notification.track_id,
                        timestamp=notification.timestamp,
                        class_name=class_name,
                        description=notification.description,
                        confidence=confidence,
                        best_area=area,
                        best_image_b64=notification.image_b64,
                        last_activity_frame=self._frame_count,
                    )

                    if self.notify_callback:
                        self.notify_callback(notification)

                    if self.verbose:
                        print(
                            f"[Frame {self._frame_count}] NEW SIGN | "
                            f"ID={track_id} | {class_name} | "
                            f"conf={confidence:.2f} | {self._get_description(class_name)}"
                        )

            # Keep the largest (clearest) crop seen for this sign.
            es = self._emitted_signs.get(class_name)
            if es is not None:
                es.last_activity_frame = self._frame_count
                es.confidence = max(es.confidence, confidence)
                if area > es.best_area:
                    new_crop = self._crop_sign(frame, (x1, y1, x2, y2))
                    if new_crop is not None:
                        es.best_area = area
                        es.best_image_b64 = new_crop
                        es.improved = True

            # Refresh class activity so the cooldown window keeps sliding while
            # the sign stays in view.
            self._last_class_activity[class_name] = now

        stale_ids = [
            tid for tid, t in self._active_tracks.items()
            if self._frame_count - t.last_seen_frame > DISPLAY_TTL
        ]
        for tid in stale_ids:
            del self._active_tracks[tid]

        # Once a notified sign has been gone for DISPLAY_TTL frames, emit the best
        # crop we saw, but only if it improved on the original.
        stale_classes = [
            cls for cls, es in self._emitted_signs.items()
            if self._frame_count - es.last_activity_frame > DISPLAY_TTL
        ]
        for cls in stale_classes:
            es = self._emitted_signs.pop(cls)
            if not es.improved:
                continue
            update = SignNotification(
                track_id=es.track_id,
                class_name=es.class_name,
                description=es.description,
                confidence=round(es.confidence, 3),
                bbox=(),
                timestamp=es.timestamp,
                frame_number=self._frame_count,
                image_b64=es.best_image_b64,
                is_update=True,
            )
            notifications.append(update)
            if self.notify_callback:
                self.notify_callback(update)
            if self.verbose:
                print(
                    f"[Frame {self._frame_count}] IMAGE UPDATE | "
                    f"ID={es.track_id} | {es.class_name} | clearer crop"
                )

        return notifications

    def process_video(
        self,
        source,
        display: bool = True,
        save_path: str = None,
    ):
        import cv2

        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video source: {source}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Pace file playback to the source fps so it doesn't look fast-forwarded.
        # A live camera already delivers frames at its own rate.
        is_live = isinstance(source, int)
        frame_interval = 1.0 / fps if fps > 0 else 0.0

        writer = None
        if save_path:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(save_path, fourcc, fps, (frame_w, frame_h))

        print(f" Started: {source} | {frame_w}x{frame_h} @ {fps:.1f}fps")

        try:
            while True:
                loop_start = time.time()
                ret, frame = cap.read()
                if not ret:
                    break

                notifications = self.process_frame(frame, frame_w, frame_h)

                if display or writer:
                    annotated = self._draw_tracks(frame.copy())
                    if writer:
                        writer.write(annotated)
                    if display:
                        cv2.imshow("Traffic Sign Detector", annotated)
                        # Wait out the rest of this frame's time budget (waitKey
                        # needs at least 1 ms).
                        if is_live:
                            wait_ms = 1
                        else:
                            remaining = frame_interval - (time.time() - loop_start)
                            wait_ms = max(1, int(remaining * 1000))
                        if cv2.waitKey(wait_ms) & 0xFF == ord("q"):
                            break

                for n in notifications:
                    yield n

        finally:
            cap.release()
            if writer:
                writer.release()
            if display:
                cv2.destroyAllWindows()

    def _draw_tracks(self, frame):
        import cv2

        for track in self._active_tracks.values():
            x1, y1, x2, y2 = [int(v) for v in track.bbox]
            color = (0, 255, 0) if track.notified else (0, 165, 255)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            label = f"ID{track.track_id} {track.class_name} {track.confidence:.2f}"
            (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (x1, y1 - lh - 6), (x1 + lw, y1), color, -1)
            cv2.putText(
                frame, label, (x1, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1
            )

        cv2.putText(
            frame,
            f"Frame: {self._frame_count} | Active tracks: {len(self._active_tracks)}",
            (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2,
        )
        return frame

    def reset(self):
        self._active_tracks.clear()
        self._notified_ids.clear()
        self._last_class_activity.clear()
        self._emitted_signs.clear()
        self._frame_count = 0
        # Rebuild the tracker so old ids and state don't leak into the next session.
        self.deepsort = self._DeepSort(**self._deepsort_kwargs)