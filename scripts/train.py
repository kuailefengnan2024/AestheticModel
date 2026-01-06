"""
模型训练脚本 (Model Training Script)

功能:
1.  加载数据集 (Train/Val Split)。
2.  初始化双塔审美模型 (AestheticScorer)。
3.  执行 Pairwise Ranking 训练循环。
4.  实时记录日志 (WandB / Console)。
5.  保存最优模型权重。

使用方法:
    python scripts/train.py --batch_size 32 --epochs 10 --lr 1e-5
"""
import os
import sys
import argparse
import random
import time
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import wandb

# 引入项目模块
sys.path.append(str(Path(__file__).resolve().parent.parent))

from config import settings
from core.dataset import AestheticDataset, collate_fn
from core.architecture import AestheticScorer, ModelConfig
from core.loss import CombinedRankingLoss
from utils.logger import logger

# ==============================================================================
# 训练配置
# ==============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Train Aesthetic Scorer")
    
    # Data
    parser.add_argument("--data_path", type=str, default="data/preferences_train.jsonl")
    parser.add_argument("--val_split", type=float, default=0.1, help="Validation set ratio")
    
    # Model
    # 默认指向 Hugging Face Hub ID
    parser.add_argument("--vision_model", type=str, default="openai/clip-vit-large-patch14", help="CLIP model name or path")
    
    # Training
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate for backbone")
    parser.add_argument("--head_lr", type=float, default=1e-4, help="Learning rate for heads (usually higher)")
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    
    # Loss
    parser.add_argument("--rank_weight", type=float, default=1.0)
    parser.add_argument("--reg_weight", type=float, default=0.1)
    
    # System
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--output_dir", type=str, default="outputs/checkpoints")
    parser.add_argument("--wandb", action="store_true", default=False, help="Enable WandB logging")
    parser.add_argument("--run_name", type=str, default=None)
    
    return parser.parse_args()

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
    full_dataset = AestheticDataset(
        data_path=args.data_path, 
        target_size=224, 
        model_name=args.vision_model
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
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    logger.info(f"Train samples: {train_size}, Val samples: {val_size}")
    
    # 2. Model
    logger.info("Initializing model...")
    # 这里不需要传 pretrained="openai"，因为 transformers 用 path 自动识别
    config = ModelConfig(
        vision_model_name=args.vision_model
    )
    model = AestheticScorer(config)
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
        
    logger.info("Training completed.")
    if args.wandb:
        wandb.finish()

if __name__ == "__main__":
    main()
