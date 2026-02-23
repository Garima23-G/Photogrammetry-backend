import argparse
import json
import mimetypes
import uuid
from pathlib import Path
from urllib import error, request


def build_multipart_form_data(fields, files):
    boundary = f"----WebKitFormBoundary{uuid.uuid4().hex}"
    lines = []

    for name, value in fields.items():
        lines.append(f"--{boundary}".encode("utf-8"))
        lines.append(f'Content-Disposition: form-data; name="{name}"'.encode("utf-8"))
        lines.append(b"")
        lines.append(str(value).encode("utf-8"))

    for field_name, file_path in files:
        filename = file_path.name
        content_type = mimetypes.guess_type(filename)[0] or "application/octet-stream"
        file_bytes = file_path.read_bytes()

        lines.append(f"--{boundary}".encode("utf-8"))
        lines.append(
            f'Content-Disposition: form-data; name="{field_name}"; filename="{filename}"'.encode(
                "utf-8"
            )
        )
        lines.append(f"Content-Type: {content_type}".encode("utf-8"))
        lines.append(b"")
        lines.append(file_bytes)

    lines.append(f"--{boundary}--".encode("utf-8"))
    lines.append(b"")

    body = b"\r\n".join(lines)
    content_type = f"multipart/form-data; boundary={boundary}"
    return body, content_type


def post_json_response(url, fields, files):
    body, content_type = build_multipart_form_data(fields=fields, files=files)
    req = request.Request(
        url=url,
        data=body,
        headers={"Content-Type": content_type},
        method="POST",
    )
    with request.urlopen(req, timeout=120) as resp:
        return resp.status, json.loads(resp.read().decode("utf-8"))


def get_index_json(base_url):
    index_url = base_url.rstrip("/") + "/"
    req = request.Request(url=index_url, method="GET")
    with request.urlopen(req, timeout=30) as resp:
        return resp.status, json.loads(resp.read().decode("utf-8"))


def main():
    parser = argparse.ArgumentParser(description="Post dummy capture files to backend API.")
    parser.add_argument("--base-url", default="http://127.0.0.1:5001", help="Backend base URL.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "dummy_capture",
        help="Folder containing metadata.json + rgb_i/depth_i files.",
    )
    args = parser.parse_args()

    input_dir = args.input_dir
    metadata_path = input_dir / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing metadata file: {metadata_path}")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    frame_count = int(metadata.get("frame_count", 0))
    if frame_count <= 0:
        raise ValueError("metadata.json must include frame_count > 0.")

    base_fields = {
        "intrinsics": json.dumps(metadata["intrinsics"]),
        "poses": json.dumps(metadata["poses"]),
    }

    sanity_url = args.base_url.rstrip("/") + "/pose_sanity"
    process_url = args.base_url.rstrip("/") + "/process"

    try:
        index_status, index_json = get_index_json(args.base_url)
        print(f"/ [{index_status}]")
        if isinstance(index_json, dict) and "endpoints" in index_json:
            print("Available endpoints:")
            for k in index_json["endpoints"].keys():
                print(f"- {k}")
    except Exception:
        # Best-effort diagnostic only.
        pass

    sanity_ok = False
    try:
        sanity_status, sanity_json = post_json_response(sanity_url, base_fields, [])
        print(f"/pose_sanity [{sanity_status}]")
        print(json.dumps(sanity_json, indent=2))
        sanity_ok = True
    except error.HTTPError as e:
        if e.code == 404:
            print("/pose_sanity not found [404].")
            print("Continuing with /process. Restart backend from latest code to enable /pose_sanity.")
        else:
            print(f"/pose_sanity failed [{e.code}]")
            print(e.read().decode("utf-8"))
            return

    process_fields = dict(base_fields)
    process_fields["category"] = metadata.get("category", "handhole")

    files = []
    for i in range(frame_count):
        rgb_path = input_dir / f"rgb_{i}.png"
        depth_path = input_dir / f"depth_{i}.png"
        if not rgb_path.exists() or not depth_path.exists():
            raise FileNotFoundError(f"Missing rgb/depth pair for frame {i} in {input_dir}")
        files.append((f"rgb_{i}", rgb_path))
        files.append((f"depth_{i}", depth_path))

    try:
        process_status, process_json = post_json_response(process_url, process_fields, files)
        print(f"/process [{process_status}]")
        print(json.dumps(process_json, indent=2))
        if not sanity_ok:
            print("NOTE: /process succeeded, but /pose_sanity is unavailable on this running backend instance.")
    except error.HTTPError as e:
        print(f"/process failed [{e.code}]")
        print(e.read().decode("utf-8"))


if __name__ == "__main__":
    main()
