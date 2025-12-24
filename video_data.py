from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch


def _require(cond: bool, msg: str):
    if not cond:
        raise RuntimeError(msg)


@dataclass(frozen=True)
class ManifestItem:
    video: str
    caption: str
    task: Optional[str] = None
    cond_image: Optional[str] = None
    fps: Optional[float] = None
    meta: Optional[Dict[str, Any]] = None


def load_manifest_jsonl(path: str | Path) -> List[ManifestItem]:
    path = Path(path)
    _require(path.exists(), f"Manifest not found: {path}")
    items: List[ManifestItem] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            items.append(
                ManifestItem(
                    video=obj["video"],
                    caption=obj["caption"],
                    task=obj.get("task"),
                    cond_image=obj.get("cond_image"),
                    fps=obj.get("fps"),
                    meta=obj.get("meta"),
                )
            )
    return items


def _load_frames_from_path(path: Path, num_frames: int):
    from PIL import Image
    import numpy as np

    if path.is_dir():
        files = sorted([p for p in path.iterdir() if p.suffix.lower() in {".jpg", ".png", ".webp"}])
        if not files:
            return [Image.new("RGB", (512, 512))] * num_frames
        indices = np.linspace(0, len(files) - 1, num_frames).astype(int)
        return [Image.open(files[i]).convert("RGB") for i in indices]

    try:
        import imageio.v3 as iio

        arr = iio.imread(path)
        if arr.ndim == 3:
            arr = arr[None]
        indices = np.linspace(0, len(arr) - 1, num_frames).astype(int)
        return [Image.fromarray(arr[i]) for i in indices]
    except Exception:
        return [Image.new("RGB", (512, 512))] * num_frames


def _process_image_bucketed(img, size: Tuple[int, int] = (512, 512)):
    import numpy as np

    img = img.resize(size)
    arr = np.array(img).astype(np.float32) / 255.0
    arr = arr * 2.0 - 1.0
    return arr.transpose(2, 0, 1)  # C, H, W


class ManifestVideoDataset(torch.utils.data.Dataset):
    def __init__(self, manifest_path: str | Path, *, num_frames: int = 16, resolution: int = 512):
        import numpy as np

        self._np = np
        self.items = load_manifest_jsonl(manifest_path)
        self.num_frames = int(num_frames)
        self.resolution = int(resolution)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        p = Path(item.video)
        frames = _load_frames_from_path(p, self.num_frames)
        tensors = [_process_image_bucketed(f, size=(self.resolution, self.resolution)) for f in frames]
        video = torch.tensor(self._np.stack(tensors)).float()  # [T, C, H, W]
        return {"video": video, "caption": item.caption}


def collate_video_batch(batch):
    return {
        "video": torch.stack([b["video"] for b in batch]),  # [B, T, C, H, W]
        "caption": [b["caption"] for b in batch],
    }

