import sys
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from pathlib import Path
from tqdm import tqdm

# Add project root
sys.path.append(str(Path(__file__).resolve().parent.parent))

from core.dataset import AestheticDataset, collate_fn
from core.architecture import AestheticScorer, ModelConfig
from core.loss import CombinedRankingLoss
from config import settings

# Force optimizations
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

def profile_training():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    # 1. Dataset
    print("Loading Dataset...")
    ds = AestheticDataset(
        data_path="data/preferences_train.jsonl",
        target_size=224,
        model_name="openai/clip-vit-large-patch14",
        image_dir="outputs/raw/pairs_512"
    )
    
    dl = DataLoader(ds, batch_size=16, shuffle=True, num_workers=0, collate_fn=collate_fn)
    
    # 2. Model
    print("Loading Model...")
    config = ModelConfig(vision_model_name="openai/clip-vit-large-patch14")
    model = AestheticScorer(config).to(device)
    model.train()
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    criterion = CombinedRankingLoss()
    
    print("\n=== Start Profiling Loop (5 Batches) ===")
    
    iter_dl = iter(dl)
    
    for i in range(5):
        print(f"\nBatch {i+1}:")
        
        # 1. Data Load
        t0 = time.time()
        try:
            batch = next(iter_dl)
        except StopIteration:
            break
        t1 = time.time()
        print(f"  [Data Load]: {t1 - t0:.4f}s")
        
        # 2. To Device
        win_pixel = batch["winner_pixel_values"].to(device)
        lose_pixel = batch["loser_pixel_values"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        gt_win = batch["scores_winner"].to(device)
        gt_lose = batch["scores_loser"].to(device)
        t2 = time.time()
        print(f"  [To Device]: {t2 - t1:.4f}s")
        
        # 3. Forward
        out_win = model(win_pixel, input_ids=input_ids, attention_mask=attention_mask)
        out_lose = model(lose_pixel, input_ids=input_ids, attention_mask=attention_mask)
        t3 = time.time()
        print(f"  [Forward]  : {t3 - t2:.4f}s")
        
        # 4. Loss
        loss, _ = criterion(out_win, out_lose, gt_win, gt_lose)
        t4 = time.time()
        print(f"  [Calc Loss]: {t4 - t3:.4f}s")
        
        # 5. Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        t5 = time.time()
        print(f"  [Backward] : {t5 - t4:.4f}s")
        
        print(f"  >> Total   : {t5 - t0:.4f}s")

if __name__ == "__main__":
    profile_training()
