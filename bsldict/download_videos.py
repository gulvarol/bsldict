import argparse
import os
import pickle as pkl
import subprocess
from pathlib import Path

from tqdm import tqdm


def download_hosted_video(video_link, output_path):
    """
        Given the link to a video, download
            using wget into output_file location.
    """
    command = ["wget", "-O", f"{output_path}", f'"{video_link}"']
    command = " ".join(command)
    try:
        output = subprocess.check_output(command, shell=True, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as err:
        print(f"Video not downloaded: {err}.")
        pass


def download_youtube_video(video_identifier, output_path):
    """
        Given the youtube video_identifier, download
            using youtube-dl into output_path location.
    """
    url_base = "https://www.youtube.com/watch?v="
    command = [
        "youtube-dl",
        f'"{url_base}{video_identifier}"',
        "-f",
        "mp4",
        "-o",
        f'"{output_path}"',
    ]
    command = " ".join(command)
    try:
        output = subprocess.check_output(command, shell=True, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as err:
        print(f"Video not downloaded: {err}.")
        pass


def main(data_path: Path):
    if not data_path.exists():
        data_path.mkdir()

    data = pkl.load(open("bsldict_v1.pkl", "rb"))

    missing_data = []
    for i, name in tqdm(enumerate(data["videos"]["name"])):
        output_path = data_path / name
        if not output_path.exists():
            # Download with wget
            if data["videos"]["download_method_db"][i] == "wget":
                download_hosted_video(data["videos"]["video_link_db"][i], output_path)
            # Download with youtube-dl
            elif data["videos"]["download_method_db"][i] == "youtube-dl":
                download_youtube_video(data["videos"]["youtube_identifier_db"][i], output_path)
            else:
                raise ValueError("Unrecognized download method")

            # Check if download succeeded
            if not output_path.exists():
                print(f"Failed to download {output_path}.")
                missing_data.append(i)

    with open("data/missing_data.txt", "w") as f:
        f.write(" ".join([str(i) + "\n" for i in missing_data]))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Script to download bsldict videos.")
    p.add_argument(
        "--data_path",
        type=Path,
        default=Path("videos_original"),
        help="path to download BSLDict videos",
    )
    main(**vars(p.parse_args()))
