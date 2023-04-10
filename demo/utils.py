import math
import os
import shutil
import sys
import time
from pathlib import Path

import cv2
import matplotlib.colors as mcolors
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import gridspec
from PIL import Image
from tqdm import tqdm

sys.path.append("..")
from bsldict.download_videos import download_hosted_video, download_youtube_video
from models.i3d_mlp import i3d_mlp
from models.i3d import InceptionI3d


def viz_similarities(
    rgb: torch.Tensor,
    t_mid: np.ndarray,
    sim: np.ndarray,
    similarity_thres: float,
    keyword: str,
    output_path: Path,
    viz_with_dict: bool,
    dict_video_links: tuple,
):
    """
    Save a visualization video for similarities
        between the input video and dictionary videos.
    1) Create a figure for every frame of the input video.
    2) On the left: show the input video frame, with the search keyword below
        (the keyword is displayed as red if below the maximum similarity is below
        similarity threshold, as green otherwise).
    3) On the right: show the plots, each corresponding to a different dictionary version,
        display the maximum similarity for the given frame.
    4) At the top: show side-by-side the middle frames of the dictionary videos
        corresponding to the keyword.
    """
    F = 16  # num_in_frames
    # Put linebreaks for long strings every 40 chars
    keyword = list(keyword)
    max_num_chars_per_line = 40
    num_linebreaks = int(len(keyword) / max_num_chars_per_line)
    for lb in range(num_linebreaks):
        pos = (lb + 1) * max_num_chars_per_line
        keyword.insert(pos, "\n")
    keyword = "".join(keyword)
    keyword = f"Keyword: {keyword}"
    num_frames = rgb.shape[1]
    height = rgb.shape[2]
    offset = height / 14
    fig = plt.figure(figsize=(9, 3))
    # 900, 300
    figw, figh = fig.get_size_inches() * fig.dpi
    figw, figh = int(figw), int(figh)
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 2])
    ax1 = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[1])
    sim_plots = ax2.plot(range(int(F / 2), int(F / 2) + sim.shape[0]), sim)
    ax2.set_ylabel("Similarity")
    ax2.set_xlabel("Time")
    ax2.set_xlim(0, num_frames - 1)
    num_versions = sim.shape[1]
    plt.legend([f"v{v + 1}" for v in range(num_versions)], loc="upper right")
    if viz_with_dict:
        res = 256
        dict_video_urls, dict_youtube_ids = dict_video_links
        num_dicts = len(dict_video_urls)
        stacked_dicts = np.zeros((num_dicts, res, res, 3))
        for v, dict_vid_url in enumerate(dict_video_urls):
            dict_color = sim_plots[v].get_color()
            # dict_color = list(mcolors.TABLEAU_COLORS.values())[0]
            yid = dict_youtube_ids[v]
            dict_frame = get_dictionary_frame(
                dict_vid_url, yid, v=f"v{v + 1}", color=dict_color, res=res
            )
            stacked_dicts[v] = dict_frame
        dict_viz = np.hstack(stacked_dicts)
        dh, dw, _ = dict_viz.shape
        dh, dw = int(figw * dh / dw), figw
        dict_viz = cv2.resize(dict_viz, (dw, dh))
    else:
        dh = 0

    # Create videowriter
    print(f"Saving visualization to {output_path}")
    FOURCC = "mp4v"
    fourcc = cv2.VideoWriter_fourcc(*FOURCC)
    out_video = cv2.VideoWriter(str(output_path), fourcc, 25, (figw, figh + dh))

    for t in tqdm(range(num_frames)):
        img = cv2.resize(im_to_numpy(rgb[:, t]), (256, 256))
        ax1.imshow(img)
        ax1.set_title("Continuous input")
        t_ix = abs(t_mid - t).argmin()
        sim_t = max(sim[t_ix, :])
        # Title
        sim_text = f"Max similarity: {sim_t:.2f}"
        ax2.set_title(sim_text)
        time_line = ax2.axvline(x=t_ix)
        time_rect = ax2.add_patch(
            patches.Rectangle(
                (t_ix, ax2.get_ylim()[0]), F, np.diff(ax2.get_ylim())[0], alpha=0.5,
            )
        )
        sim_color = "red"
        if sim_t >= similarity_thres:
            sim_color = "green"
            # Rectangle whenever above a sim thres
            ax1.add_patch(
                patches.Rectangle(
                    (0, 0), 256, 256, linewidth=10, edgecolor="g", facecolor="none"
                )
            )
        # Display keyword
        ax1.text(
            offset,
            256,
            keyword,
            fontsize=12,
            fontweight="bold",
            color="white",
            verticalalignment="top",
            bbox=dict(facecolor=sim_color, alpha=0.9),
        )
        ax1.axis("off")
        if t == 0:
            plt.tight_layout()
        fig_img = fig2data(fig)
        fig_img = np.array(Image.fromarray(fig_img))
        if viz_with_dict:
            fig_img = np.vstack((dict_viz, fig_img))
        out_video.write(fig_img[:, :, (2, 1, 0)].astype("uint8"))
        # cv2.imwrite(f"frames/frame_{t:04d}.png", fig_img[:, :, (2, 1, 0)].astype("uint8"))
        ax1.clear()
        time_line.remove()
        time_rect.remove()
    out_video.release()
    msg = (f"Did not find a generated video at {output_path}, is the FOURCC {FOURCC} "
           f"supported by your opencv install?")
    assert output_path.exists(), msg


def fig2data(fig):
    """
    @brief Convert a Matplotlib figure to a 3D numpy array with RGB channels and return it
    @param fig a matplotlib figure
    @return a numpy 3D array of RGB values
    """
    # draw the renderer
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep="")
    buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return buf


def get_dictionary_frame(url, yid, v, color, res=256, rm_download=True):
    """
    1) Download the dictionary video to a temporary file location
        (with youtube-dl if it has a youtube-identifier, otherwise with wget),
        if cannot download, return black image,
    2) Read the middle frame of the video,
    3) Resize to a square resolution determined by `res`,
    4) Put a rectangle around the frame with `color` (convert matplotlib color to cv2 color),
    5) Put a text to display the dictionary version number (determined by `v`)
    6) Convert BGR to RGB and return the frame
    7) Remove the temporary download

    :param url: video url
    :param yid: youtube_identifier (can be None)
    :param v: string to display on top of the frame, denotes the dictionary version number
    :param color: color in matplotlib style, to be used in the rectangle and text
    :param res: resolution of the frame at which to resize
    """
    try:
        # Temporary file location
        # tmp = f"tmp_{time.time()}.mp4"
        tmp = f"tmp-{v}.mp4"
        if yid:
            # Download with youtube-dl
            download_youtube_video(yid, tmp)
        else:
            # Download with wget
            download_hosted_video(url, tmp)
        # Read the video
        cap = cv2.VideoCapture(tmp)
        # Get the total number of frames
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        # Read the middle frame
        cap.set(propId=1, value=math.floor(frame_count / 2))
        ret, frame = cap.read()
        frame = cv2.resize(frame, (res, res))
        # Version color (matplotlib=>cv2)
        rgb_color = mcolors.to_rgb(color)
        rgb_color = tuple((255 * rgb_color[2], 255 * rgb_color[1], 255 * rgb_color[0]))
        # Rectangle frame
        frame = cv2.rectangle(frame, (0, 0), (res - 1, res - 1), rgb_color, 20)
        # Version text
        frame = cv2.rectangle(frame, (30, 20), (80, 60), rgb_color, -1)
        frame = cv2.putText(
            frame, v, (40, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 4
        )
        # BGR => RGB
        frame = frame[:, :, [2, 1, 0]]
        if rm_download:
            # Remove temporary download
            os.remove(tmp)
        return frame
    except:
        print(f"Could not download dictionary video {url}")
        return np.zeros((res, res, 3))


def load_rgb_video(video_path: Path, fps: int) -> torch.Tensor:
    """
    Load frames of a video using cv2 (fetch from provided URL if file is not found
    at given location).
    """
    cap = cv2.VideoCapture(str(video_path))
    cap_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cap_fps = cap.get(cv2.CAP_PROP_FPS)

    # cv2 won't be able to change frame rates for all encodings, so we use ffmpeg
    if cap_fps != fps:
        tmp_video_path = f"{video_path}.tmp.{video_path.suffix}"
        shutil.move(video_path, tmp_video_path)
        cmd = (
            f"ffmpeg -i {tmp_video_path} -pix_fmt yuv420p "
            f"-filter:v fps=fps={fps} {video_path}"
        )
        print(f"Generating new copy of video with frame rate {fps}")
        os.system(cmd)
        Path(tmp_video_path).unlink()
        cap = cv2.VideoCapture(str(video_path))
        cap_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        cap_fps = cap.get(cv2.CAP_PROP_FPS)
        assert cap_fps == fps, f"ffmpeg failed to produce a video at {fps}"

    f = 0
    rgb = []
    while True:
        # frame: BGR, (h, w, 3), dtype=uint8 0..255
        ret, frame = cap.read()
        if not ret:
            break
        # BGR (OpenCV) to RGB (Torch)
        frame = frame[:, :, [2, 1, 0]]
        rgb_t = im_to_torch(frame)
        rgb.append(rgb_t)
        f += 1
    cap.release()
    # (nframes, 3, cap_height, cap_width) => (3, nframes, cap_height, cap_width)
    rgb = torch.stack(rgb).permute(1, 0, 2, 3)
    print(
        f"Loaded video {video_path} with {f} frames [{cap_height}hx{cap_width}w] res. "
        f"at {cap_fps}"
    )
    return rgb


def prepare_input(
    rgb: torch.Tensor,
    resize_res: int = 256,
    inp_res: int = 224,
    mean: torch.Tensor = 0.5 * torch.ones(3),
    std=1.0 * torch.ones(3),
):
    """
    Process the video:
    1) Resize to [resize_res x resize_res]
    2) Center crop with [inp_res x inp_res]
    3) Color normalize using mean/std
    """
    iC, iF, iH, iW = rgb.shape
    # Resize
    rgb_resized = np.zeros((iF, resize_res, resize_res, iC))
    for t in range(iF):
        tmp = rgb[:, t, :, :]
        rgb_resized[t] = cv2.resize(im_to_numpy(tmp), (resize_res, resize_res))

    rgb = np.transpose(rgb_resized, (3, 0, 1, 2))
    # Center crop coords
    ulx = int((resize_res - inp_res) / 2)
    uly = int((resize_res - inp_res) / 2)
    # Crop 256x256
    rgb = rgb[:, :, uly : uly + inp_res, ulx : ulx + inp_res]
    rgb = to_torch(rgb).float()
    assert rgb.max() <= 1
    rgb = color_normalize(rgb, mean, std)
    return rgb


def load_model(checkpoint_path: Path, arch: str) -> torch.nn.Module:
    """Load pre-trained checkpoint, put in eval mode.
    """
    if arch == "i3d_mlp":
        model = i3d_mlp()
    elif arch == "i3d":
        model = InceptionI3d(num_classes=2281, num_in_frames=16, include_embds=True)
    else:
        raise ValueError(f"Unrecognized architecture {arch}")
    checkpoint = torch.load(str(checkpoint_path))
    if arch == "i3d":
        model = torch.nn.DataParallel(model)  # .cuda()
        checkpoint = checkpoint["state_dict"]
    model.load_state_dict(checkpoint)
    model.eval()
    return model


def sliding_windows(rgb: torch.Tensor, num_in_frames: int, stride: int,) -> tuple:
    """
    Return sliding windows and corresponding (middle) timestamp
    """
    C, nFrames, H, W = rgb.shape
    # If needed, pad to the minimum clip length
    if nFrames < num_in_frames:
        rgb_ = torch.zeros(C, num_in_frames, H, W)
        rgb_[:, :nFrames] = rgb
        rgb_[:, nFrames:] = rgb[:, -1].unsqueeze(1)
        rgb = rgb_
        nFrames = rgb.shape[1]

    num_clips = math.ceil((nFrames - num_in_frames) / stride) + 1
    plural = ""
    if num_clips > 1:
        plural = "s"
    print(f"{num_clips} clip{plural} resulted from sliding window processing.")

    rgb_slided = torch.zeros(num_clips, 3, num_in_frames, H, W)
    t_mid = []
    # For each clip
    for j in range(num_clips):
        # Check if num_clips becomes 0
        actual_clip_length = min(num_in_frames, nFrames - j * stride)
        if actual_clip_length == num_in_frames:
            t_beg = j * stride
        else:
            t_beg = nFrames - num_in_frames
        t_mid.append(t_beg + num_in_frames / 2)
        rgb_slided[j] = rgb[:, t_beg : t_beg + num_in_frames, :, :]
    return rgb_slided, np.array(t_mid)


def im_to_numpy(img):
    img = to_numpy(img)
    img = np.transpose(img, (1, 2, 0))  # H*W*C
    return img


def im_to_torch(img):
    img = np.transpose(img, (2, 0, 1))  # C*H*W
    img = to_torch(img).float()
    if img.max() > 1:
        img /= 255
    return img


def to_numpy(tensor):
    if torch.is_tensor(tensor):
        return tensor.cpu().numpy()
    elif type(tensor).__module__ != "numpy":
        raise ValueError(f"Cannot convert {type(tensor)} to numpy array")
    return tensor


def to_torch(ndarray):
    if type(ndarray).__module__ == "numpy":
        return torch.from_numpy(ndarray)
    elif not torch.is_tensor(ndarray):
        raise ValueError(f"Cannot convert {type(ndarray)} to torch tensor")
    return ndarray


def color_normalize(x, mean, std):
    """Normalize a tensor of images by subtracting the mean, dividing by std. dev.
    """
    if x.dim() in {3, 4}:
        if x.size(0) == 1:
            x = x.repeat(3, 1, 1)
        assert x.size(0) == 3, "For single video format, expected RGB along first dim"
        for t, m, s in zip(x, mean, std):
            t.sub_(m)
            t.div_(s)
    elif x.dim() == 5:
        assert (
            x.shape[1] == 3
        ), "For batched video format, expected RGB along second dim"
        x[:, 0].sub_(mean[0]).div_(std[0])
        x[:, 1].sub_(mean[1]).div_(std[1])
        x[:, 2].sub_(mean[2]).div_(std[2])
    return x


def viz_slide(
    rgb: torch.Tensor,
    t_mid: np.ndarray,
    sim: np.ndarray,
    similarity_thres: float,
    keyword: str,
    output_path: Path,
    viz_with_dict: bool,
    dict_video_links: tuple,
    max_num_versions: int = 1,
):
    viz_with_dict = 0
    """
    Save a visualization video for talks, rearranged version of the function viz_similarities above
    """
    F = 16  # num_in_frames
    # Put linebreaks for long strings every 40 chars
    keyword = list(keyword)
    max_num_chars_per_line = 40
    num_linebreaks = int(len(keyword) / max_num_chars_per_line)
    for lb in range(num_linebreaks):
        pos = (lb + 1) * max_num_chars_per_line
        keyword.insert(pos, "\n")
    keyword = "".join(keyword)
    keyword = f"Keyword: {keyword}"
    num_frames = rgb.shape[1]
    height = rgb.shape[2]
    offset = height / 14
    dict_video_urls, dict_youtube_ids = dict_video_links
    num_versions = sim.shape[1]
    ## delete this hack
    # sim = sim[:, (3, 0, 1, 2, 3)]
    ##
    if num_versions > max_num_versions:
        sim = sim[:, :max_num_versions]
        num_versions = sim.shape[1]
        dict_video_urls = dict_video_urls[:max_num_versions]
        dict_youtube_ids = dict_youtube_ids[:max_num_versions]
    fig = plt.figure(figsize=(6, 3 + num_versions * 3))
    # 900, 300
    figw, figh = fig.get_size_inches() * fig.dpi
    figw, figh = int(figw), int(figh)
    gs = gridspec.GridSpec(num_versions + 1, 1, height_ratios=[3] + num_versions * [1])
    ax1 = plt.subplot(gs[0])
    res = 256
    num_dicts = len(dict_video_urls)
    stacked_dicts = np.zeros((num_dicts, res, res, 3))
    assert num_dicts == num_versions
    for v, dict_vid_url in enumerate(dict_video_urls):
        dict_color = list(mcolors.TABLEAU_COLORS.values())[v]
        ax2 = plt.subplot(gs[v + 1])
        sim_plot = ax2.plot(range(int(F / 2), int(F / 2) + sim.shape[0]), sim[:, v], color=dict_color)
        # ax2.set_ylabel("Similarity")
        ax2.set_xlabel("Time")
        ax2.set_xlim(0, num_frames - 1)
        ax2.set_ylim(sim.min(), sim.max() + 0.01)
        # plt.savefig(f"plot-{v+1}.png")
        if num_versions > 1:
            plt.legend([f"v{v + 1}"], loc="upper right")
            plt.savefig(f"plot-{v+1}-withlegend.png")

        # dict_color = sim_plot[0].get_color()
        yid = dict_youtube_ids[v]
        dict_frame = get_dictionary_frame(
            dict_vid_url, yid, v=f"v{v + 1}", color=dict_color, res=res, rm_download=False
        )
        stacked_dicts[v] = dict_frame
    if viz_with_dict:
        dict_viz = np.vstack(stacked_dicts)
        dh, dw, _ = dict_viz.shape
        # dh, dw = int(figw * dh / dw), figw
        dh, dw = figh, int(figh * dw / dh)
        dict_viz = cv2.resize(dict_viz, (dw, dh))
    else:
        # dh = 0
        dw = 0

    # Create videowriter
    print(f"Saving visualization to {output_path}")
    FOURCC = "mp4v"
    fourcc = cv2.VideoWriter_fourcc(*FOURCC)
    out_video = cv2.VideoWriter(str(output_path), fourcc, 25, (figw + dw, figh))

    for t in tqdm(range(num_frames)):
        img = cv2.resize(im_to_numpy(rgb[:, t]), (256, 256))
        ax1.imshow(img)
        ax1.set_title("Continuous input")
        t_ix = abs(t_mid - t).argmin()
        time_lines = []
        time_rects = []
        for v in range(num_dicts):
            sim_t = sim[t_ix, v]
            # Title
            sim_text = f"Similarity: {sim_t:.2f}"
            ax2 = plt.subplot(gs[v + 1])
            ax2.set_title(sim_text)
            right_frame = ax2.spines["right"]
            right_frame.set_visible(False)
            top_frame = ax2.spines["top"]
            top_frame.set_visible(False)
            time_line = ax2.axvline(x=t_ix)
            time_rect = ax2.add_patch(
                patches.Rectangle(
                    (t_ix, ax2.get_ylim()[0]), F, np.diff(ax2.get_ylim())[0], alpha=0.5,
                )
            )
            time_lines.append(time_line)
            time_rects.append(time_rect)
        sim_color = "red"
        max_sim_t = max(sim[t_ix, :])
        if max_sim_t >= similarity_thres:
            sim_color = "green"
            # Rectangle whenever above a sim thres
            ax1.add_patch(
                patches.Rectangle(
                    (0, 0), 256, 256, linewidth=10, edgecolor="g", facecolor="none"
                )
            )
        # Display keyword
        ax1.text(
            offset,
            256,
            keyword,
            fontsize=12,
            fontweight="bold",
            color="white",
            verticalalignment="top",
            bbox=dict(facecolor=sim_color, alpha=0.9),
        )
        ax1.axis("off")
        if t == 0:
            plt.tight_layout()
        fig_img = fig2data(fig)
        fig_img = np.array(Image.fromarray(fig_img))
        if viz_with_dict:
            fig_img = np.hstack((dict_viz, fig_img))
        out_video.write(fig_img[:, :, (2, 1, 0)].astype("uint8"))
        cv2.imwrite(f"frames/frame_{t:04d}.png", fig_img[:, :, (2, 1, 0)].astype("uint8"))
        ax1.clear()
        for v in range(num_dicts):
            time_lines[v].remove()
            time_rects[v].remove()
    out_video.release()
    msg = (f"Did not find a generated video at {output_path}, is the FOURCC {FOURCC} "
           f"supported by your opencv install?")
    assert output_path.exists(), msg
