import argparse
import sys
from pathlib import Path

from youtube_core import download_video


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download a YouTube video from a share URL."
    )
    parser.add_argument("url", help="YouTube share URL, e.g. https://youtu.be/VIDEO_ID")
    parser.add_argument(
        "--resolution",
        type=int,
        default=720,
        help="Maximum height in pixels (default: 720)",
    )
    parser.add_argument(
        "--out",
        default="downloads",
        help="Output folder (default: downloads)",
    )
    parser.add_argument(
        "--route",
        default="uncategorized",
        help="Route folder name under the output directory (default: uncategorized)",
    )

    args = parser.parse_args()

    try:
        result = download_video(
            args.url,
            Path(args.out),
            args.resolution,
            route_folder=args.route,
        )
        print(f"Download complete: {result.video_path}")
        return 0
    except Exception as exc:
        print(f"Download failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
