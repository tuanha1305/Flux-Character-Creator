from pathlib import Path
from typing import List, Dict, Tuple, Any
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from transformers import CLIPTokenizer


class GameCharacterComponentDataset(Dataset):
    def __init__(self,
                 root_dir: str,
                 component: str,
                 tokenizer: CLIPTokenizer,
                 max_length: int = 77,
                 image_size: Tuple[int, int] = (512, 512)):
        self.root_dir = Path(root_dir)
        self.component = component
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.image_size = image_size

        self.image_paths = self._get_image_paths()
        self.transform = self._get_transforms()

    def _get_image_paths(self) -> List[Path]:
        component_dir = self.root_dir / self.component
        return list(component_dir.glob("*.png"))

    def _get_transforms(self):
        return transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGBA")

        # Convert transparent background to white for training
        background = Image.new("RGBA", image.size, (255, 255, 255))
        image = Image.alpha_composite(background, image).convert("RGB")

        image = self.transform(image)

        caption_path = img_path.with_suffix(".txt")
        with open(caption_path, "r") as f:
            caption = f.read().strip()

        encoded_caption = self.tokenizer(
            caption,
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt"
        )

        return {
            "pixel_values": image,
            "input_ids": encoded_caption["input_ids"].squeeze(0),
            "attention_mask": encoded_caption["attention_mask"].squeeze(0)
        }

    def get_component_info(self) -> Dict[str, Any]:
        return {
            "component": self.component,
            "num_samples": len(self),
            "image_size": self.image_size
        }


def create_component_datasets(root_dir: str,
                              components: List[str],
                              tokenizer: CLIPTokenizer) -> Dict[str, GameCharacterComponentDataset]:
    return {
        component: GameCharacterComponentDataset(root_dir, component, tokenizer)
        for component in components
    }
