import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from pathlib import Path
import json
from PIL import Image
from einops import rearrange
from tqdm import tqdm
import wandb

from video_dataset_process import VideoTextDataset, collate_fn
from video_generation_model import TitansVideoGenerator, TitansVideoTrainer



def train():
    """训练脚本"""
    # 配置
    config = {
        'data_root': '/path/to/video/dataset',  # 修改为你的数据路径
        'output_dir': './outputs',
        'num_frames': 16,
        'resolution': 512,
        'batch_size': 1,  # 根据显存调整
        'num_epochs': 100,
        'learning_rate': 1e-4,
        'num_workers': 2,
        'save_every': 1000,
        'log_every': 10,
        'gradient_accumulation_steps': 4,
        'mixed_precision': True,
    }

    # 创建输出目录
    output_dir = Path(config['output_dir'])
    output_dir.mkdir(exist_ok=True, parents=True)

    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 数据集
    dataset = VideoTextDataset(
        data_root=config['data_root'],
        video_length=config['num_frames'],
        resolution=config['resolution'],
    )

    dataloader = DataLoader(
        dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        collate_fn=collate_fn,
        pin_memory=True,
    )

    # 模型
    reasoning_config = {
        'num_physics_rules': 128,
        'num_scenes': 256,
        'num_entities': 64,
        'memory_depth': 3,
        'reasoning_depth': 4,
        'consistency_depth': 3,
    }

    model = TitansVideoGenerator(
        reasoning_config=reasoning_config
    ).to(device)

    # 训练器
    trainer = TitansVideoTrainer(
        model=model,
        learning_rate=config['learning_rate'],
        use_8bit_adam=True,
        gradient_accumulation_steps=config['gradient_accumulation_steps'],
    )

    # 训练循环
    global_step = 0

    print("开始训练...")
    for epoch in range(config['num_epochs']):
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{config['num_epochs']}")

        for batch_idx, batch in enumerate(pbar):
            # 移动到设备
            video = batch['video'].to(device)
            text = batch['text']
            images = batch['image'].to(device) if batch['image'] is not None else None

            # 训练步骤
            losses = trainer.train_step(video, text, images)

            # 更新进度条
            if global_step % config['log_every'] == 0:
                pbar.set_postfix(losses)

            # 保存检查点
            if global_step % config['save_every'] == 0 and global_step > 0:
                checkpoint_path = output_dir / f'checkpoint-{global_step}.pt'
                torch.save({
                    'step': global_step,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': trainer.optimizer.state_dict(),
                    'config': config,
                }, checkpoint_path)

                print(f"\n保存检查点到 {checkpoint_path}")

            global_step += 1

    print("训练完成！")

def save_video(video: torch.Tensor, output_path: str, fps: int = 8):
    """保存视频"""
    try:
        import imageio
        # [T, 3, H, W] -> [T, H, W, 3]
        video = rearrange(video, 't c h w -> t h w c')
        video = (video * 255).cpu().numpy().astype(np.uint8)
        imageio.mimwrite(output_path, video, fps=fps)

    except Exception as e:
        print(f"保存视频失败: {e}")

def inference():
    """推理脚本"""
    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载模型
    print("加载模型...")
    model = TitansVideoGenerator().to(device)
    trainer = TitansVideoTrainer(model)

    # 加载检查点
    checkpoint_path = 'outputs/checkpoint-10000.pt'
    if Path(checkpoint_path).exists():
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"加载检查点: {checkpoint_path}")
    else:
        print("未找到检查点，使用随机初始化权重")

    # 文本提示
    prompts = [
        "A cat walking in a garden, realistic, 4k, beautiful lighting",
        "Ocean waves crashing on a beach at sunset, cinematic",
    ]

    print("生成视频...")
    videos = trainer.generate(
        text=prompts,
        num_frames=16,
        height=512,
        width=512,
        num_inference_steps=50,
        guidance_scale=7.5,
    )

    # 保存视频
    for i, video in enumerate(videos):
        output_path = f'output_video_{i}.mp4'
        save_video(video, output_path)
        print(f"保存视频到 {output_path}")



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Titans视频生成')
    parser.add_argument('--mode', type=str, default='inference',
                        choices=['train', 'inference'],
                        help='运行模式')

    args = parser.parse_args()

    if args.mode == 'train':
        train()
    else:
        inference()