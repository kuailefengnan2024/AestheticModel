import os
import argparse
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import multiprocessing
from functools import partial

def resize_image(file_path, output_dir, size=512):
    """
    Resizes an image so its longest side is at most `size`, maintaining aspect ratio.
    """
    try:
        dest_path = output_dir / file_path.name
        
        # Skip if already exists
        if dest_path.exists():
            return
            
        with Image.open(file_path) as img:
            # Convert to RGB to handle PNGs with alpha channel or other modes
            img = img.convert("RGB")
            
            w, h = img.size
            if max(w, h) > size:
                scale = size / max(w, h)
                new_w = int(w * scale)
                new_h = int(h * scale)
                img = img.resize((new_w, new_h), Image.BICUBIC)
            
            img.save(dest_path, quality=95)
    except Exception as e:
        print(f"Error processing {file_path}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Batch resize images for training")
    parser.add_argument("--input_dir", type=str, default="outputs/raw/pairs", help="Source directory")
    parser.add_argument("--output_dir", type=str, default="outputs/raw/pairs_512", help="Target directory")
    parser.add_argument("--size", type=int, default=512, help="Max dimension size")
    parser.add_argument("--workers", type=int, default=8, help="Number of worker processes")
    
    args = parser.parse_args()
    
    input_path = Path(args.input_dir)
    output_path = Path(args.output_dir)
    
    if not input_path.exists():
        print(f"Input directory not found: {input_path}")
        return

    output_path.mkdir(parents=True, exist_ok=True)
    
    # Get all image files
    files = list(input_path.glob("*.png")) + list(input_path.glob("*.jpg")) + list(input_path.glob("*.jpeg"))
    print(f"Found {len(files)} images in {input_path}")
    print(f"Resizing to max {args.size}px -> {output_path}...")
    
    # Process in parallel
    process_func = partial(resize_image, output_dir=output_path, size=args.size)
    
    with multiprocessing.Pool(args.workers) as pool:
        list(tqdm(pool.imap_unordered(process_func, files), total=len(files)))
        
    print("Done!")

if __name__ == "__main__":
    main()
