"""Usage examples:
    python main.py --source test_videos/my_video.mp4
    python main.py --source 0                          # webcam
    python main.py --source test_videos/test.mp4 --save output/result.mp4 --no-display
"""

import argparse
import json
import time
from pathlib import Path

from tracker import SignTracker, SignNotification


def print_notification(n: SignNotification):
    print(
        f"\n{'='*55}\n"
        f"   NEW SIGN DETECTED\n"
        f"   Description: {n.description}\n"
        f"   Class      : {n.class_name}\n"
        f"   Track ID   : {n.track_id}\n"
        f"   Confidence : {n.confidence:.1%}\n"
        f"   Frame      : {n.frame_number}\n"
        f"{'='*55}"
    )


def main():
    parser = argparse.ArgumentParser(description="Traffic Sign Detector local test")
    parser.add_argument(
        "--source", type=str, default="0",
        help="Video file or camera index (0 for webcam)"
    )
    parser.add_argument(
        "--model", type=str, default="models/best.pt",
        help="Path to YOLO model"
    )
    parser.add_argument(
        "--conf", type=float, default=0.45,
        help="Minimum confidence threshold"
    )
    parser.add_argument(
        "--save", type=str, default=None,
        help="Path to save annotated video (e.g. output/result.mp4)"
    )
    parser.add_argument(
        "--no-display", action="store_true",
        help="Don't show window (useful for headless server)"
    )
    parser.add_argument(
        "--log", type=str, default=None,
        help="Save notifications to JSON file (e.g. output/log.json)"
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Verbose console output"
    )
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Model not found: {model_path}")
        print("   Place best.pt in the models/ folder.")
        return

    source = args.source
    try:
        source = int(source)
    except ValueError:
        if not Path(source).exists():
            print(f"Video file not found: {source}")
            return

    print(f"Model: {model_path}")
    print(f"Source: {'camera ' + str(source) if isinstance(source, int) else source}")
    print(f"Confidence: {args.conf}")
    print("   Press Q to quit\n")

    all_notifications = []
    start_time = time.time()

    tracker = SignTracker(
        model_path=str(model_path),
        conf_threshold=args.conf,
        tracker_config="tracker/deepsort.yaml",
        verbose=args.verbose,
    )

    try:
        for notification in tracker.process_video(
            source=source,
            display=not args.no_display,
            save_path=args.save,
        ):
            print_notification(notification)
            all_notifications.append({
                "track_id": notification.track_id,
                "class_name": notification.class_name,
                "description": notification.description,
                "confidence": notification.confidence,
                "bbox": notification.bbox,
                "frame": notification.frame_number,
                "timestamp": notification.timestamp,
            })

    except KeyboardInterrupt:
        print("\nInterrupted by user.")

    elapsed = time.time() - start_time
    print(f"\nTotal signs detected: {len(all_notifications)}")
    print(f"Duration: {elapsed:.1f}s")

    if args.log and all_notifications:
        log_path = Path(args.log)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(all_notifications, f, ensure_ascii=False, indent=2)
        print(f"Log saved: {log_path}")


if __name__ == "__main__":
    main()
