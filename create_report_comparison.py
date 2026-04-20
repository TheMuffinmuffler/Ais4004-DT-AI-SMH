
import os
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path

def create_comparison_grid(target_gauge="gauge_1", param_to_compare="st"):
    plots_dir = Path("plots")
    # Find folders that match our criteria (win100, bs32, Eps50, Lr0.001)
    # We want to see how 'st' (stride) changes things.
    folders = [d for d in plots_dir.iterdir() if d.is_dir() and "win100" in d.name and "bs32" in d.name and "Eps50" in d.name]
    
    images = []
    labels = []
    
    for folder in sorted(folders):
        # Extract the stride value for the label
        parts = folder.name.split("_")
        stride_val = [p for p in parts if p.startswith("st")][0]
        
        img_path = folder / f"comparison_{target_gauge}.png"
        if img_path.exists():
            images.append(Image.open(img_path))
            labels.append(f"Stride: {stride_val[2:]}")

    if not images:
        print("No matching plots found for comparison.")
        return

    # Create a side-by-side montage
    widths, heights = zip(*(i.size for i in images))
    total_width = sum(widths)
    max_height = max(heights)

    new_im = Image.new('RGB', (total_width, max_height + 50), (255, 255, 255))
    
    x_offset = 0
    draw = ImageDraw.Draw(new_im)
    
    for i, img in enumerate(images):
        new_im.paste(img, (x_offset, 50))
        # Draw Label
        draw.text((x_offset + 10, 10), labels[i], fill=(0, 0, 0))
        x_offset += img.size[0]

    output_path = f"comparison_{target_gauge}_by_stride.png"
    new_im.save(output_path)
    print(f"Comparison saved to: {output_path}")

if __name__ == "__main__":
    create_comparison_grid()
