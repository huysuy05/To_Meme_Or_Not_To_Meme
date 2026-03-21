"""
This file helps load the metadata from KYM (Know Your Meme) and ImgFlip
"""

from pathlib import Path
import random
from typing import List
import matplotlib.pyplot as plt
from PIL import Image


class DataLoader:
    def __init__(self, root_dir: str):
        self.root_dir = Path(root_dir)
        if not self.root_dir.exists() or not self.root_dir.is_dir():
            raise ValueError(f"Invalid dataset path: {self.root_dir}")

        self.valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".gif"}

    def get_template_names(self) -> list[str]:
        """Return all subfolder names as a list of strings."""
        return [d.name for d in self._template_dirs()]

    def _template_dirs(self) -> List[Path]:
        return sorted([p for p in self.root_dir.iterdir() if p.is_dir()])

    def _images_in_dir(self, d: Path) -> List[Path]:
        return sorted([p for p in d.iterdir() if p.is_file() and p.suffix.lower() in self.valid_exts])

    def count_templates(self) -> int:
        """Count number of subfolders (templates)."""
        return len(self._template_dirs())

    def display_templates(self, n: int = 10) -> List[str]:
        """Print and return the first n subfolders."""
        dirs = self._template_dirs()[:n]
        names = [d.name for d in dirs]
        print(f"First {len(names)} template folders:")
        for i, name in enumerate(names, 1):
            print(f"{i:02d}. {name}")
        return names

    def display_tpl_image(self, max_images: int = 9, seed: int = None):
        """
        Pick a random subfolder and display images inside it.
        Returns (template_name, image_paths).
        """
        dirs = self._template_dirs()
        if not dirs:
            raise ValueError("No template subfolders found.")

        rng = random.Random(seed)
        chosen_dir = rng.choice(dirs)
        images = self._images_in_dir(chosen_dir)

        if not images:
            print(f"Template '{chosen_dir.name}' has no image files.")
            return chosen_dir.name, []

        show_imgs = images[:max_images]
        cols = min(3, len(show_imgs))
        rows = (len(show_imgs) + cols - 1) // cols

        plt.figure(figsize=(4 * cols, 4 * rows))
        for i, img_path in enumerate(show_imgs, 1):
            img = Image.open(img_path)
            plt.subplot(rows, cols, i)
            plt.imshow(img)
            plt.title(img_path.name, fontsize=9)
            plt.axis("off")
        plt.suptitle(f"Random template: {chosen_dir.name}", fontsize=12)
        plt.tight_layout()
        plt.show()

        return chosen_dir.name, [str(p) for p in images]
    


if __name__ == "__main__":
    dataset_path = "/Volumes/huysuy05/ssd_data/meme_or_not/IMGFlip2024_haslabel"
    loader = DataLoader(dataset_path)

    print("Total templates:", loader.count_templates())
    loader.display_templates(10)
    template_name, image_list = loader.display_tpl_image(max_images=9)
    print(f"Selected template: {template_name}")
    print(f"Total images in selected template: {len(image_list)}")
    names = loader.get_template_names()
    print(names[:10])
