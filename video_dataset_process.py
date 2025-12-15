import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from pathlib import Path
import json
import numpy as np
from PIL import Image
from einops import rearrange, repeat
from typing import Optional, Tuple, List
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

try:
    from diffusers import AutoencoderKL, UNet3DConditionModel
    from diffusers.schedulers import DDPMScheduler
    from transformers import CLIPTextModel, CLIPTokenizer, CLIPVisionModel, CLIPImageProcessor
except ImportError:
    print("请先安装依赖: pip install diffusers transformers accelerate")
    exit(1)


# class VideoTextDataset(Dataset):
#     """视频-文本数据集"""
#     def __init__(
#             self,
#             data_root: str,
#             video_length: int = 16,
#             resolution: int = 512,
#             sample_fps: int = 8,
#     ):
#         self.data_root = Path(data_root)
#         self.video_length = video_length
#         self.resolution = resolution
#         self.sample_fps = sample_fps
#
#         # 加载元数据
#         with open(self.data_root / "metadata.json") as f:
#             self.metadata = json.load(f)
#
#         self.transform = transforms.Compose([
#             transforms.Resize((resolution, resolution)),
#             transforms.ToTensor(),
#             transforms.Normalize([0.5], [0.5]),
#         ])
#
#     def __len__(self):
#         return len(self.metadata)
#
#     def __getitem__(self, idx):
#         item = self.metadata[idx]
#
#         # 加载视频帧
#         video_path = self.data_root / item['video_path']
#         frames = []
#
#         # 均匀采样帧
#         import cv2
#         cap = cv2.VideoCapture(str(video_path))
#         total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#         frame_indices = torch.linspace(
#             0, total_frames - 1, self.video_length
#         ).long()
#
#         for frame_idx in frame_indices:
#             cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
#             ret, frame = cap.read()
#             if ret:
#                 frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#                 frame = Image.fromarray(frame)
#                 frame = self.transform(frame)
#                 frames.append(frame)
#
#         cap.release()
#
#         video = torch.stack(frames, dim=0)  # [T, 3, H, W]
#         video = rearrange(video, 't c h w -> t c h w')
#
#         # 文本提示
#         text = item['caption']
#
#         # 可选：参考图像（第一帧）
#         reference_image = frames[0] if 'use_first_frame' in item else None
#
#         return {
#             'video': video,
#             'text': text,
#             'image': reference_image,
#         }
#
#
# def collate_fn(batch):
#     """批处理函数"""
#     videos = torch.stack([item['video'] for item in batch])
#     texts = [item['text'] for item in batch]
#
#     images = None
#     if batch[0]['image'] is not None:
#         images = torch.stack([item['image'] for item in batch])
#
#     return {
#         'video': videos,
#         'text': texts,
#         'image': images,
#     }



class VideoTextDataset(Dataset):
    """视频-文本数据集"""
    def init(
            self,
            data_root: str,
            video_length: int = 16,
            resolution: int = 512,
            sample_fps: int = 8,
    ):
        self.data_root = Path(data_root)
        self.video_length = video_length
        self.resolution = resolution
        self.sample_fps = sample_fps

        # 加载元数据
        metadata_path = self.data_root / "metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"找不到元数据文件: {metadata_path}")

        with open(metadata_path) as f:
            self.metadata = json.load(f)

        print(f"加载了 {len(self.metadata)} 个视频样本")

        # 图像变换
        self.transform = transforms.Compose([
            transforms.Resize((resolution, resolution)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        item = self.metadata[idx]

        # 加载视频
        video_path = self.data_root / item['video_path']

        try:
            import cv2
            frames = self._load_video_cv2(video_path)
        except Exception as e:
            print(f"加载视频失败 {video_path}: {e}")
            # 返回随机数据作为占位符
            return self._get_placeholder_data()

        video = torch.stack(frames, dim=0)  # [T, 3, H, W]

        # 文本提示
        text = item.get('caption', 'a video')

        # 可选：参考图像（用于图生视频）
        reference_image = frames[0] if item.get('use_first_frame', False) else None

        return {
            'video': video,
            'text': text,
            'image': reference_image,
        }

    def _load_video_cv2(self, video_path):
        """使用OpenCV加载视频"""
        import cv2

        cap = cv2.VideoCapture(str(video_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # 均匀采样帧
        if total_frames < self.video_length:
            frame_indices = list(range(total_frames))
            # 重复最后一帧
            frame_indices += [total_frames - 1] * (self.video_length - total_frames)
        else:
            frame_indices = torch.linspace(
                0, total_frames - 1, self.video_length
            ).long().tolist()

        frames = []
        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()

            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = Image.fromarray(frame)
                frame = self.transform(frame)
                frames.append(frame)

        cap.release()

        return frames

    def _get_placeholder_data(self):
        """返回占位符数据"""
        video = torch.randn(self.video_length, 3, self.resolution, self.resolution)
        return {
            'video': video,
            'text': 'placeholder',
            'image': video[0],
        }

def collate_fn(batch):
        """批处理函数"""
        videos = torch.stack([item['video'] for item in batch])
        texts = [item['text'] for item in batch]
        images = None
        if batch[0]['image'] is not None:
            images = torch.stack([item['image'] for item in batch])

        return {
            'video': videos,
            'text': texts,
            'image': images,
        }