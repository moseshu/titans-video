# Titans video

```bash
# 推理（生成视频）
python titans_video_generation.py --mode inference

# 训练
python titans_video_generation.py --mode train
```

**注意事项：**

1. **依赖安装：**
```bash
pip install torch torchvision diffusers transformers accelerate imageio opencv-python einops
```

2. **数据准备：** 需要准备 `metadata.json` 格式如：
```json
[
  {
    "video_path": "videos/video1.mp4",
    "caption": "A cat playing in the garden"
  }
]
```

3. **显存优化：** 如果显存不足，可以：
    - 减小 `batch_size`
    - 减小 `resolution`
    - 减小 `num_frames`
    - 启用梯度检查点