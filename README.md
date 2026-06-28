# Traffic Sign Detection and Tracking

Real-time traffic sign detection, tracking and audio announcement system. A
fine-tuned YOLO model detects signs in a video stream, DeepSORT tracks each sign
across frames so it is announced only once, and a mobile app shows the detections
and plays a spoken description for the driver.

The project has three parts:

- **Detection + tracking core** (`tracker/`, `main.py`) – YOLO inference wrapped
  with DeepSORT and the notification logic. Runs standalone on a video file or
  webcam.
- **Backend** (`backend/`) – a FastAPI WebSocket server that streams frames from
  the phone, runs the core pipeline, and returns detections (with a cropped
  image of each detected sign).
- **Mobile app** (`mobile/`) – an Expo / React Native app that streams the
  camera (or a demo video) to the backend, lists detected signs, and plays an
  audio cue for each one.

## How it works

1. The model runs on every frame and outputs candidate sign boxes above a
   confidence threshold (default **0.45**).
2. Boxes that are too small (far away / not yet legible) are dropped before
   tracking, so a sign is only classified once it is close enough.
3. DeepSORT assigns a stable track ID to each physical sign across frames. A sign
   is reported **once per track**, when its track is confirmed.
4. A class-level cooldown collapses duplicates that happen when DeepSORT briefly
   loses a sign and re-acquires it under a new track ID.
5. The first alert fires while the sign is still far and the crop is blurry; as
   the car approaches, the largest / sharpest crop is kept and pushed to the app
   as an image update.

The 50 supported sign classes and their descriptions are defined in
`tracker/sign_tracker.py`.

## Project structure

```
main.py                     Standalone CLI runner (video file or webcam)
tracker/
  sign_tracker.py           YOLO + DeepSORT pipeline and notification logic
  deepsort.yaml             Tracker and filtering parameters
backend/
  main.py                   FastAPI WebSocket server
  test_client.py            Streams a local video to the backend (no phone needed)
models/
  best.pt                   Fine-tuned YOLO weights used for inference
notebooks/
  fine-tuning-yolo26n.ipynb Training / fine-tuning notebook
mobile/                     Expo / React Native app
```

## Requirements

- Python 3.10+
- Node.js 18+ and the Expo CLI (for the mobile app)
- Dependencies in `requirements.txt`

```bash
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

## Running

### Standalone (video file or webcam)

```bash
python main.py --source test_videos/my_video.mp4
python main.py --source 0                              # webcam
python main.py --source test_videos/test.mp4 --save output/result.mp4 --no-display
```

Useful flags: `--conf` (confidence threshold), `--log` (save detections to JSON),
`--verbose`.

### Backend

```bash
python backend/main.py
# or
uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

Verify the full pipeline without the phone by streaming a local video:

```bash
python backend/test_client.py --video test_videos/my_video.mp4
```

### Mobile app

```bash
cd mobile
npm install
```

Set the backend address in `mobile/.env` (see `.env.example`):

```
EXPO_PUBLIC_BACKEND_URL=<your-computer-ip>:8000
```

Then build and run it on a connected Android device:

```bash
npx expo run:android
```

The app relies on native modules (camera, audio, video), so it cannot run in
Expo Go – it must be built with `npx expo run:android`. After the first native
build you can use `npx expo start` for subsequent runs.

The app has a **Live** mode (phone camera) and a **Demo video** mode (pick a
video from the device). Detected signs appear in the list with the cropped image,
and an audio description plays for each new sign.

## Model

The model is a fine-tuned YOLO detector trained on a traffic-sign dataset (see
`notebooks/fine-tuning-yolo26n.ipynb`). The trained weights live in
`models/best.pt`.
