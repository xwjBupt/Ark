"""Study-level datasets. No external medical data or test-time fitting."""
import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset
from torchvision import transforms as T

CLASSES = [
    "Enlarged Cardiomediastinum", "Pneumothorax", "Consolidation",
    "Pneumonia", "Edema", "Cardiomegaly", "Atelectasis", "Lung Opacity",
    "Pleural Effusion", "No Finding",
]
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def canonical_image(path, canvas=512):
    """Lossless cache and raw-file inference use exactly the same letterboxing."""
    with Image.open(path) as source:
        image = source.convert("L")
        image = ImageOps.contain(image, (canvas, canvas), Image.Resampling.BICUBIC)
        output = Image.new("L", (canvas, canvas), 0)
        output.paste(image, ((canvas-image.width)//2, (canvas-image.height)//2))
    return output


def build_transform(size, training=False):
    operations = [T.Resize((size, size), interpolation=T.InterpolationMode.BICUBIC)]
    if training:
        operations += [
            T.RandomAffine(7, translate=(0.025, 0.025), scale=(0.94, 1.06),
                           interpolation=T.InterpolationMode.BILINEAR, fill=0),
            T.ColorJitter(brightness=0.12, contrast=0.12),
        ]
    operations += [T.Grayscale(num_output_channels=3), T.ToTensor(), T.Normalize(MEAN, STD)]
    return T.Compose(operations)


class StudyDataset(Dataset):
    def __init__(self, manifest, size, training=False, max_views=2, raw=False):
        self.records = json.loads(Path(manifest).read_text())
        self.transform = build_transform(size, training)
        self.training, self.max_views, self.raw = training, max_views, raw

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        images = record["images"]
        if self.training and len(images) > self.max_views:
            frontal = [i for i in images if i["view"] == 1]
            lateral = [i for i in images if i["view"] == 2]
            if frontal and lateral:
                selected = [random.choice(frontal), random.choice(lateral)]
                remaining = [i for i in images if i not in selected]
                images = selected + random.sample(remaining, min(len(remaining), self.max_views-2))
            else:
                images = random.sample(images, self.max_views)
        tensors = [self.transform(canonical_image(i["raw"] if self.raw else i["cache"])) for i in images]
        labels = record.get("labels", [0]*10)
        return torch.stack(tensors), torch.tensor([i["view"] for i in images]), torch.tensor(labels, dtype=torch.float32), index


def collate_studies(batch):
    lengths = [len(item[0]) for item in batch]
    return (
        torch.cat([item[0] for item in batch]),
        torch.cat([item[1] for item in batch]),
        torch.repeat_interleave(torch.arange(len(batch)), torch.tensor(lengths)),
        torch.stack([item[2] for item in batch]),
        torch.tensor([item[3] for item in batch]),
    )


def seed_worker(worker_id):
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)
