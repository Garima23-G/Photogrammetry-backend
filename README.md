# Mobile Capture Backend: Reconstruction & Measurement Pipeline

This repository contains the backend implementation for a mobile-based asset capture system. It processes sequential RGBD data and camera poses to reconstruct 3D scenes and perform automated geometric measurements for various infrastructure assets.

## Pipeline Overview

The pipeline consists of four main stages:

### Stage 04: Reconstruction
Performs RGBD volumetric fusion using **Open3D ScalableTSDFVolume**.
- **Bypasses SfM/COLMAP:** Uses pre-computed metric poses.
- **Voxel Size:** 4mm (0.004m) for sub-centimeter accuracy.
- **Deliverables:** Metric point clouds and TriangleMeshes with recomputed normals.

### Stage 05: Asset Measurement Engine
Applies category-specific geometric algorithms on the reconstructed point cloud:
- **Trench:** Calculates depth via Z-range and width via multi-slice averaging.
- **Manhole:** Fits a RANSAC circle to the top slice to determine diameter.
- **Duct:** Robust arc fitting and PCA axis projection for diameter and length.
- **Handhole:** Uses Minimal Oriented Bounding Box (OBB) for accurate L x W x H dimensions regardless of installation angle.

### Stage 06: Accuracy Validation
Validated accuracy targets:
- Trench Width: ±1–3%
- Manhole Diameter: <1%
- Duct Diameter: ±2–4%
- Handhole Dimensions: ±1–2%

### Stage 07: End-to-End API
A Flask-based API that wires the capture app to the reconstruction and measurement logic.

## Project Structure

```text
├── api/                # Flask API implementation
├── measurement/        # Stage 05 measurement algorithms
├── reconstruction/     # Stage 04 volumetric fusion logic
├── tests/              # Comprehensive unit and E2E tests
├── requirements.txt    # Python dependencies
└── README.md           # Documentation
```

## Setup & Installation

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **System Requirements:**
   - Python 3.8+
   - Open3D
   - NumPy
   - OpenCV
   - Flask

## Running the API

Start the development server:
```bash
export FLASK_APP=api/app.py
flask run --host=0.0.0.0 --port=5000
```

The API expects a `POST` request to `/process` with `multipart/form-data`.

## Running Tests

Execute the full test suite including unit tests, accuracy validation, and E2E integration:

```bash
python3 -m pytest tests/
```

To see accuracy reports during testing:
```bash
python3 -m pytest tests/accuracy_validation.py -s
```
