from typing import Union

import clip
import ffmpeg
import numpy as np
import torch
from clip.model import CLIP

clip_model = {}


class Normalize(object):
    def __init__(
        self,
        mean: Union[tuple, list] = (0.48145466, 0.4578275, 0.40821073),
        std: Union[tuple, list] = (0.26862954, 0.26130258, 0.27577711),
        eps: float = 1e-12,
    ):
        self.mean = torch.tensor(mean).view(3, 1, 1)
        self.std = torch.tensor(std).view(3, 1, 1)
        self.eps = torch.tensor(eps)

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized Tensor image.
        """
        tensor = tensor / 255.0
        return (tensor - self.mean) / torch.max(self.std, self.eps)


normalize = Normalize()


def extract_frames(
    video_path: str,
    frame_rate=1,
    resolution: Union[tuple, int] = 224,
    center_crop: bool = True,
) -> np.ndarray:
    """
    Extract frames from a video file.
    Args:
        video_path (str): Path to the video file.
        frame_rate (int): Frame rate to extract frames.
        resolution (tuple or int): Resolution of the extracted frames.
            If int, the smaller side of the video will be resized to this value.
        center_crop (bool): Whether to center crop the frames.
    Returns:
        frames (list): List of extracted frames.
    """
    probe = ffmpeg.probe(video_path)
    video_info = next(
        stream for stream in probe["streams"] if stream["codec_type"] == "video"
    )
    width = int(video_info["width"])
    height = int(video_info["height"])

    if type(resolution) == int:
        if width > height:
            resolution = (resolution * width // height, resolution)
        else:
            resolution = (resolution, resolution * height // width)
    min_side = min(resolution)

    stream = ffmpeg.input(video_path)
    stream = ffmpeg.filter(stream, "fps", fps=frame_rate)
    stream = ffmpeg.filter(stream, "scale", w=resolution[0], h=resolution[1])

    if center_crop:
        stream = ffmpeg.filter(
            stream,
            "crop",
            w=min_side,
            h=min_side,
            x=(width - min_side) // 2,
            y=(height - min_side) // 2,
        )
        resolution = (min_side, min_side)
    stream = ffmpeg.output(stream, "pipe:", format="rawvideo", pix_fmt="rgb24")
    out, _ = ffmpeg.run(stream, capture_stdout=True)
    frames = np.frombuffer(out, np.uint8).reshape([-1, resolution[1], resolution[0], 3])
    return frames


def extract_features(
    frames: np.ndarray,
    model: Union[CLIP, str] = "ViT-L/14",
) -> np.ndarray:
    """
    Extract features from frames using CLIP.
    Args:
        frames (list): List of frames.
        model (clip.CLIP): CLIP model.
    Returns:
        features (list): List of extracted features.
    """
    if type(model) == str:
        if model not in clip_model:
            clip_model[model], _ = clip.load(model)
        model = clip_model[model]
    frames = np.transpose(frames, (0, 3, 1, 2))
    frames = torch.from_numpy(frames).float()
    frames = normalize(frames)

    with torch.no_grad():
        features = model.encode_image(frames)
    return features.cpu().numpy()
