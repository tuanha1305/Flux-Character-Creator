from pathlib import Path
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


class GameCharacterDataset(Dataset):
    def __init__(self, root_dir, tokenizer, max_length=77):
        self.root_dir = Path(root_dir)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.image_paths = list(self.root_dir.glob("*.png"))
        self.transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        caption_path = img_path.with_suffix(".txt")
        with open(caption_path, "r") as f:
            caption = f.read().strip()

        example = self.tokenizer(
            caption,
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt"
        )
        example["pixel_values"] = image
        return example
