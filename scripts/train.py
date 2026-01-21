"""
模型训练脚本 (Model Training Script)

性能优化提示 (Performance Tips):
1. 硬件加速 (Hardware):
   - 必须开启 TF32 (TensorFloat-32)，否则 RTX 3090/4090 上 CLIP ViT-L 前向极慢 (3.6s vs 0.06s)。
   - 本脚本已自动检测并开启 `allow_tf32` 和 `cudnn.benchmark`。

2. 显存管理 (VRAM):
   - 必须开启 Gradient Checkpointing，否则 ViT-L 极易爆 24GB 显存，导致 Swap Thrashing (速度骤降 30x)。
   - 推荐 Batch Size <= 12 (为了稳定默认设为 8)。
   - 本脚本已默认开启 `gradient_checkpointing_enable`。

3. 数据加载 (DataLoader):
   - Windows 下必须设置 `num_workers=0`，否则 Spawn 进程开销巨大且会导致代码重复执行。
   - Linux 下可设置 `num_workers=4`。

⚠️ 重要提示:
   - 在 Windows/IDE 中关闭终端窗口并不会真正杀死训练进程！
   - 进程仍会在后台占用显存。
   - 必须打开 [任务管理器] -> [详细信息] -> 右键列头启用 [专用 GPU 内存] -> 找到 python.exe 且显存占用高的进程 -> [结束任务]。

功能:
1.  加载数据集 (Train/Val Split)。
2.  初始化双塔审美模型 (AestheticScorer)。
3.  执行 Pairwise Ranking 训练循环。
4.  实时记录日志 (WandB / Console)。
5.  保存最优模型权重。

使用方法:
    python scripts/train.py --batch_size 32 --epochs 10 --lr 1e-5
"""
import warnings
# 过滤无关紧要的 Pydantic 警告
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

import os
import sys
import argparse
import random
import time
import warnings
from pathlib import Path
from datetime import datetime

# Filter pydantic warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import wandb
import matplotlib.pyplot as plt

# 引入项目模块
sys.path.append(str(Path(__file__).resolve().parent.parent))

from config import settings
from core.dataset import AestheticDataset, collate_fn
from core.architecture import AestheticScorer, ModelConfig
from core.loss import CombinedRankingLoss
from utils.logger import logger

# ==============================================================================
# 默认配置区域 (Default Configuration)
# 如果不想使用命令行参数，可以在这里直接修改
# ==============================================================================
CONFIG = {
    # Data
    "data_path": "data/preferences_train.jsonl",
    "val_split": 0.1,
    
    # Model
    "vision_model": "openai/clip-vit-large-patch14",
    "freeze_backbone": True, # CRITICAL: Freeze CLIP for small dataset (<10k pairs)
    "dropout": 0.5, # High dropout to prevent overfitting
    
    # Training
    "batch_size": 32, # Safe to increase if backbone is frozen (grads only on heads)
    "epochs": 50,
    "lr": 1e-5, # Unused for backbone if frozen
    "head_lr": 1e-3, # Higher LR for heads
    "weight_decay": 0.1, # Strong regularization
    "grad_clip": 1.0,
    
    # Loss
    "rank_weight": 1.0,
    "reg_weight": 0.1,
    
    # System
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    # Windows 下建议设为 0，避免多进程 Spawn 开销巨大的问题
    "num_workers": 0 if os.name == 'nt' else 4,
    "output_dir": "outputs/checkpoints",
    "image_dir": "outputs/raw/pairs_512", # Default to resized images
    "wandb": False,
    "plot_loss": True,
    "run_name": None
}

# ==============================================================================
# 硬件加速 (RTX 3090/4090 必须开启)
# 移入 main 函数或在 if name == main 中调用，避免多进程重复执行
# ==============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Train Aesthetic Scorer")
    
    # Data
    parser.add_argument("--data_path", type=str, default=CONFIG["data_path"])
    parser.add_argument("--val_split", type=float, default=CONFIG["val_split"])
    
    # Model
    parser.add_argument("--vision_model", type=str, default=CONFIG["vision_model"])
    parser.add_argument("--freeze_backbone", action="store_true", default=CONFIG["freeze_backbone"])
    parser.add_argument("--dropout", type=float, default=CONFIG["dropout"])

    # Training
    parser.add_argument("--batch_size", type=int, default=CONFIG["batch_size"])
    parser.add_argument("--epochs", type=int, default=CONFIG["epochs"])
    parser.add_argument("--lr", type=float, default=CONFIG["lr"])
    parser.add_argument("--head_lr", type=float, default=CONFIG["head_lr"])
    parser.add_argument("--weight_decay", type=float, default=CONFIG["weight_decay"])
    parser.add_argument("--grad_clip", type=float, default=CONFIG["grad_clip"])
    
    # Loss
    parser.add_argument("--rank_weight", type=float, default=CONFIG["rank_weight"])
    parser.add_argument("--reg_weight", type=float, default=CONFIG["reg_weight"])
    
    # System
    parser.add_argument("--device", type=str, default=CONFIG["device"])
    parser.add_argument("--num_workers", type=int, default=CONFIG["num_workers"])
    parser.add_argument("--output_dir", type=str, default=CONFIG["output_dir"])
    parser.add_argument("--image_dir", type=str, default=CONFIG["image_dir"], help="Path to image directory (overrides jsonl paths)")
    parser.add_argument("--wandb", action="store_true", default=CONFIG["wandb"])
    parser.add_argument("--plot_loss", action="store_true", default=CONFIG["plot_loss"])
    parser.add_argument("--run_name", type=str, default=CONFIG["run_name"])
    
    return parser.parse_args()

def plot_training_curve(history, save_dir):
    """绘制并保存训练曲线"""
    try:
        epochs = range(1, len(history["train_loss"]) + 1)
        
        plt.figure(figsize=(12, 5))
        
        # Subplot 1: Loss
        plt.subplot(1, 2, 1)
        plt.plot(epochs, history["train_loss"], label="Train Loss", marker='o')
        plt.plot(epochs, history["val_loss"], label="Val Loss", marker='o')
        plt.title("Loss Curve")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)
        
        # Subplot 2: Accuracy
        plt.subplot(1, 2, 2)
        plt.plot(epochs, history["train_acc"], label="Train Acc", marker='o')
        plt.plot(epochs, history["val_acc"], label="Val Acc", marker='o')
        plt.title("Accuracy Curve")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(save_dir / "training_curve.png")
        plt.close()
        logger.info(f"Loss curve saved to {save_dir / 'training_curve.png'}")
    except Exception as e:
        logger.error(f"Failed to plot curve: {e}")

# ==============================================================================
# 核心循环
# ==============================================================================

def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch):
    model.train()
    total_loss = 0.0
    correct_pairs = 0
    total_pairs = 0
    
    pbar = tqdm(dataloader, desc=f"Train Epoch {epoch}")
    
    for batch in pbar:
        # 1. Move to device
        # Winner
        win_pixel = batch["winner_pixel_values"].to(device)
        # Loser
        lose_pixel = batch["loser_pixel_values"].to(device)
        
        # Text
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        
        # Labels
        gt_win_scores = batch["scores_winner"].to(device) # [B, 4]
        gt_lose_scores = batch["scores_loser"].to(device) # [B, 4]
        
        # 2. Forward
        # 训练时我们复用同一个模型两次 (Siamese)
        
        # Winner Pass
        out_win = model(win_pixel, input_ids=input_ids, attention_mask=attention_mask)
        # Loser Pass
        out_lose = model(lose_pixel, input_ids=input_ids, attention_mask=attention_mask)
        
        # 3. Loss
        loss, loss_dict = criterion(out_win, out_lose, gt_win_scores, gt_lose_scores)
        
        # 4. Backward
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient Clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        optimizer.step()
        
        # 5. Metrics
        total_loss += loss.item()
        
        # 计算 Total Score 的排序准确率
        if "total" in out_win:
            pred_win = out_win["total"]
            pred_lose = out_lose["total"]
            correct = (pred_win > pred_lose).sum().item()
            correct_pairs += correct
            total_pairs += len(pred_win)
            
        # Logging
        pbar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{correct_pairs/max(1, total_pairs):.2%}"})
        
        if wandb.run:
            wandb.log({
                "train/batch_loss": loss.item(), 
                "train/batch_acc": correct_pairs/max(1, total_pairs),
                **loss_dict
            })
            
    avg_loss = total_loss / len(dataloader)
    avg_acc = correct_pairs / max(1, total_pairs)
    return avg_loss, avg_acc

@torch.no_grad()
def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct_pairs = 0
    total_pairs = 0
    
    for batch in tqdm(dataloader, desc="Validating"):
        # 1. Move to device
        win_pixel = batch["winner_pixel_values"].to(device)
        lose_pixel = batch["loser_pixel_values"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        gt_win_scores = batch["scores_winner"].to(device)
        gt_lose_scores = batch["scores_loser"].to(device)
        
        # 2. Forward
        out_win = model(win_pixel, input_ids=input_ids, attention_mask=attention_mask)
        out_lose = model(lose_pixel, input_ids=input_ids, attention_mask=attention_mask)
        
        # 3. Loss
        loss, _ = criterion(out_win, out_lose, gt_win_scores, gt_lose_scores)
        total_loss += loss.item()
        
        # 4. Metrics
        if "total" in out_win:
            pred_win = out_win["total"]
            pred_lose = out_lose["total"]
            correct = (pred_win > pred_lose).sum().item()
            correct_pairs += correct
            total_pairs += len(pred_win)
            
    avg_loss = total_loss / len(dataloader)
    avg_acc = correct_pairs / max(1, total_pairs)
    return avg_loss, avg_acc

def main():
    args = parse_args()

    # ==============================================================================
    # 硬件加速 (RTX 3090/4090 必须开启)
    # 原因: CLIP ViT-L 在 FP32 下极慢 (3.6s/it)，开启 TF32 调用 Tensor Cores 后提速至 0.06s/it。
    # ==============================================================================
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        logger.info("⚡ Hardware acceleration enabled: TF32 + CuDNN Benchmark")
        
    # Setup Output Dir
    run_id = args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = Path(args.output_dir) / run_id
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup WandB
    if args.wandb:
        wandb.init(project="AestheticModel", name=run_id, config=vars(args))
    
    logger.info(f"Training started. Device: {args.device}, Save to: {save_dir}")
    
    # 1. Dataset
    logger.info("Loading dataset...")
    
    # Check if resized image dir exists, otherwise fallback to None (use jsonl paths)
    image_dir = args.image_dir
    if image_dir and not os.path.exists(image_dir):
        logger.warning(f"Image directory {image_dir} not found. Falling back to original paths in JSONL.")
        image_dir = None
        
    full_dataset = AestheticDataset(
        data_path=args.data_path, 
        target_size=224, 
        model_name=args.vision_model,
        image_dir=image_dir
    )
    
    # Split
    val_size = int(len(full_dataset) * args.val_split)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        persistent_workers=(args.num_workers > 0) # Avoid respawning workers on Windows
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        persistent_workers=(args.num_workers > 0)
    )
    
    logger.info(f"Train samples: {train_size}, Val samples: {val_size}")
    
    # 2. Model
    logger.info("Initializing model...")
    # 这里不需要传 pretrained="openai"，因为 transformers 用 path 自动识别
    config = ModelConfig(
        vision_model_name=args.vision_model,
        freeze_backbone=args.freeze_backbone,
        dropout=args.dropout
    )
    model = AestheticScorer(config)
    
    # Enable Gradient Checkpointing to save VRAM (only if backbone is trainable)
    if not args.freeze_backbone:
        if hasattr(model.clip.vision_model, "gradient_checkpointing_enable"):
            model.clip.vision_model.gradient_checkpointing_enable()
            logger.info("✅ Gradient Checkpointing enabled for Vision Model (VRAM saving)")
    else:
        logger.info("❄️ Backbone is frozen. Gradient Checkpointing disabled (not needed).")
        
    model.to(args.device)
    
    # 3. Optimizer
    # 差分学习率
    backbone_params = []
    head_params = []
    for name, param in model.named_parameters():
        if "heads" in name:
            head_params.append(param)
        else:
            backbone_params.append(param)
            
    optimizer = torch.optim.AdamW([
        {'params': backbone_params, 'lr': args.lr},
        {'params': head_params, 'lr': args.head_lr}
    ], weight_decay=args.weight_decay)
    
    # 4. Loss
    criterion = CombinedRankingLoss(rank_weight=args.rank_weight, reg_weight=args.reg_weight)
    
    # 5. Loop
    best_val_acc = 0.0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    
    for epoch in range(1, args.epochs + 1):
        logger.info(f"Epoch {epoch}/{args.epochs}")
        
        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, args.device, epoch
        )
        logger.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2%}")
        
        # Val
        val_loss, val_acc = validate(model, val_loader, criterion, args.device)
        logger.info(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2%}")
        
        # Record History
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        
        # Log
        if args.wandb:
            wandb.log({
                "epoch": epoch,
                "train/loss": train_loss,
                "train/acc": train_acc,
                "val/loss": val_loss,
                "val/acc": val_acc
            })
            
        # Save Best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            logger.info(f"New best model! (Acc: {val_acc:.2%})")
            torch.save(model.state_dict(), save_dir / "best_model.pth")
            
        # Save Last
        torch.save(model.state_dict(), save_dir / "last_model.pth")
    
    # Plot Curve
    if args.plot_loss:
        plot_training_curve(history, save_dir)
        
    logger.info("Training completed.")
    if args.wandb:
        wandb.finish()

if __name__ == "__main__":
    main()
