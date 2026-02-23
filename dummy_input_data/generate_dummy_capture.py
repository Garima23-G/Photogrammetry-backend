import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def build_dummy_payload():
    intrinsic_matrix = np.array(
        [
            [500.0, 0.0, 320.0],
            [0.0, 500.0, 240.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    poses = [np.eye(4, dtype=np.float64), np.eye(4, dtype=np.float64)]
    poses[1][0, 3] = 0.10

    h, w = 480, 640
    depth = np.zeros((h, w), dtype=np.uint16)
    depth[100:380, 200:440] = 2000

    rgb_0 = np.full((h, w, 3), 220, dtype=np.uint8)
    rgb_1 = np.full((h, w, 3), 200, dtype=np.uint8)
    cv2.rectangle(rgb_0, (200, 100), (440, 380), (30, 130, 240), 3)
    cv2.rectangle(rgb_1, (210, 110), (450, 390), (30, 180, 180), 3)

    return intrinsic_matrix, poses, [rgb_0, rgb_1], [depth, depth]


def main():
    parser = argparse.ArgumentParser(description="Generate dummy RGBD API input files.")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "dummy_capture",
        help="Output folder for generated files.",
    )
    args = parser.parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    intrinsics, poses, rgbs, depths = build_dummy_payload()

    for i, rgb in enumerate(rgbs):
        cv2.imwrite(str(out_dir / f"rgb_{i}.png"), rgb)
    for i, depth in enumerate(depths):
        cv2.imwrite(str(out_dir / f"depth_{i}.png"), depth)

    metadata = {
        "intrinsics": intrinsics.tolist(),
        "poses": [p.tolist() for p in poses],
        "category": "handhole",
        "frame_count": len(rgbs),
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Dummy capture written to: {out_dir}")
    print("Files:")
    print("- metadata.json")
    for i in range(len(rgbs)):
        print(f"- rgb_{i}.png")
        print(f"- depth_{i}.png")


if __name__ == "__main__":
    main()
