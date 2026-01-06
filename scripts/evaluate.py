"""
模型评估与推理脚本 (Evaluation & Inference)

功能:
1.  **Batch Eval**: 加载测试集，计算 Ranking Accuracy。
2.  **Single Inference**: 给定单张图片路径和 Prompt，输出模型打分。

使用方法:
    # 评估数据集
    python scripts/evaluate.py --model_path outputs/checkpoints/best_model.pth --mode batch

    # 单张推理
    python scripts/evaluate.py --model_path outputs/checkpoints/best_model.pth --mode single --image_path test.jpg --prompt "a beautiful sunset"
"""
import argparse
import json
import sys
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

# 引入项目模块
sys.path.append(str(Path(__file__).resolve().parent.parent))

from core.dataset import AestheticDataset, collate_fn, DynamicResizePad
from core.architecture import AestheticScorer, ModelConfig
import open_clip

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to .pth checkpoint")
    parser.add_argument("--mode", type=str, choices=["batch", "single"], default="batch")
    
    # Batch Args
    parser.add_argument("--data_path", type=str, default="data/preferences_train.jsonl")
    parser.add_argument("--batch_size", type=int, default=32)
    
    # Single Args
    parser.add_argument("--image_path", type=str)
    parser.add_argument("--prompt", type=str, default="")
    
    # Model Args
    parser.add_argument("--vision_model", type=str, default="ViT-L-14")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    
    return parser.parse_args()

def load_model(args):
    """加载模型和权重"""
    print(f"Loading model from {args.model_path}...")
    config = ModelConfig(vision_model_name=args.vision_model)
    model = AestheticScorer(config)
    
    # Load Weights
    checkpoint = torch.load(args.model_path, map_location=args.device)
    model.load_state_dict(checkpoint)
    model.to(args.device)
    model.eval()
    return model

def evaluate_batch(model, args):
    """批量评估准确率"""
    print(f"Evaluating on {args.data_path}...")
    dataset = AestheticDataset(args.data_path, target_size=224, model_name=args.vision_model)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=collate_fn, num_workers=4)
    
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader):
            # Move to device
            win_pixel = batch["winner_pixel_values"].to(args.device)
            lose_pixel = batch["loser_pixel_values"].to(args.device)
            input_ids = batch["input_ids"].to(args.device)
            
            # Forward
            out_win = model(win_pixel, input_ids)
            out_lose = model(lose_pixel, input_ids)
            
            # Metric: Total Score Accuracy
            if "total" in out_win:
                s_win = out_win["total"]
                s_lose = out_lose["total"]
                correct += (s_win > s_lose).sum().item()
                total += len(s_win)
                
    acc = correct / max(1, total)
    print(f"\nFinal Accuracy: {acc:.2%}")
    return acc

def inference_single(model, args):
    """单张图片推理"""
    if not args.image_path:
        print("Error: --image_path is required for single mode.")
        return

    # 1. Preprocess Image
    preprocessor = DynamicResizePad(target_size=224)
    try:
        image = Image.open(args.image_path).convert("RGB")
    except Exception as e:
        print(f"Error loading image: {e}")
        return
        
    processed = preprocessor(image)
    pixel_values = processed["pixel_values"].unsqueeze(0).to(args.device) # [1, 3, H, W]
    
    # 2. Preprocess Text
    tokenizer = open_clip.get_tokenizer(args.vision_model)
    input_ids = tokenizer(args.prompt).to(args.device) # [1, 77]
    
    # 3. Forward
    with torch.no_grad():
        outputs = model(pixel_values, input_ids)
        
    # 4. Print Results
    print("\n" + "="*30)
    print("Aesthetic Scores:")
    print("="*30)
    for k, v in outputs.items():
        print(f"{k.ljust(15)}: {v.item():.4f}")
    print("="*30)

def main():
    args = parse_args()
    model = load_model(args)
    
    if args.mode == "batch":
        evaluate_batch(model, args)
    else:
        inference_single(model, args)

if __name__ == "__main__":
    main()
