# Dummy Input Data

This folder provides synthetic RGBD capture input so you can validate backend endpoints quickly.

## Files
- `generate_dummy_capture.py`: creates a sample payload in `dummy_capture/`.
- `post_dummy_capture.py`: posts the sample payload to `/pose_sanity` and `/process`.

## Quick Start
1. Start backend:
```bash
python -m flask --app api.app run --host 0.0.0.0 --port 5000
```

2. Generate sample files:
```bash
python dummy_input_data/generate_dummy_capture.py
```

3. Post to API:
```bash
python dummy_input_data/post_dummy_capture.py --base-url http://127.0.0.1:5000
```

## Generated Payload Schema
- `metadata.json`: includes `intrinsics`, `poses`, `category`, `frame_count`.
- `rgb_0.png`, `rgb_1.png`, ...
- `depth_0.png`, `depth_1.png`, ... (`uint16` depth in millimeters)
