"""Pretrained encoder with class-specific attention over all images in a study."""
from pathlib import Path
import timm
import torch
from torch import nn

PRETRAINS = {
    "swin_base": {
        "architecture": "swin_base_patch4_window7_224",
        "filename": "swin_base_patch4_window7_224_22kto1k.pth",
        "url": "https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window7_224_22kto1k.pth",
        "license": "MIT",
        "license_url": "https://raw.githubusercontent.com/microsoft/Swin-Transformer/main/LICENSE",
        "training_data": "ImageNet-22K pretraining, ImageNet-1K fine-tuning",
        "documentation": "https://github.com/microsoft/Swin-Transformer/blob/main/MODELHUB.md",
    },
    "convnext_small": {
        "architecture": "convnext_small",
        "filename": "convnext_small_22k_1k_384.pth",
        "url": "https://dl.fbaipublicfiles.com/convnext/convnext_small_22k_1k_384.pth",
        "license": "MIT",
        "license_url": "https://raw.githubusercontent.com/facebookresearch/ConvNeXt/main/LICENSE",
        "training_data": "ImageNet-22K pretraining, ImageNet-1K fine-tuning at 384px",
        "documentation": "https://github.com/facebookresearch/ConvNeXt#results-and-pre-trained-models",
    },
}


class MultiViewModel(nn.Module):
    def __init__(self, backbone="swin_base", size=384, pretrained_path=None, drop_path=0.15,
                 init="ark", pooling="noisy_or"):
        super().__init__()
        if pooling not in {"attention", "noisy_or"}:
            raise ValueError(f"Unknown multi-label pooling: {pooling}")
        self.pooling = pooling
        kwargs = {"img_size": size} if backbone.startswith("swin") else {}
        self.encoder = timm.create_model(PRETRAINS[backbone]["architecture"], pretrained=False,
                                         num_classes=0, drop_path_rate=drop_path, **kwargs)
        if pretrained_path:
            self.load_pretrained(pretrained_path, backbone, init)
        dim = self.encoder.num_features
        self.view_embedding = nn.Embedding(3, dim)
        nn.init.zeros_(self.view_embedding.weight)
        self.dropout = nn.Dropout(0.2)
        self.classifier = nn.Linear(dim, 10)
        self.attention = nn.Linear(dim, 10)
        nn.init.zeros_(self.attention.weight)
        nn.init.zeros_(self.attention.bias)

    def load_pretrained(self, path, backbone, init):
        state = torch.load(path, map_location="cpu", weights_only=False)
        if init == "ark":
            state = state.get("teacher", state)
            state = {k.removeprefix("module."): v for k, v in state.items()}
            state = {k: v for k, v in state.items() if k.startswith(("patch_embed.", "layers.", "norm."))}
        elif init == "imagenet":
            if Path(path).name != PRETRAINS[backbone]["filename"]:
                raise ValueError("ImageNet initialization requires the documented public weight filename")
        else:
            raise ValueError(f"Unknown initialization: {init}")
        if backbone.startswith("swin"):
            from timm.models.swin_transformer import checkpoint_filter_fn
        else:
            from timm.models.convnext import checkpoint_filter_fn
        state = checkpoint_filter_fn(state, self.encoder)
        state = {k: v for k, v in state.items() if k not in {"head.fc.weight", "head.fc.bias"}}
        message = self.encoder.load_state_dict(state, strict=False)
        if message.missing_keys or message.unexpected_keys:
            raise RuntimeError(f"Incomplete pretrained encoder initialization: {message}")
        print(f"Loaded complete {backbone} {init} encoder from {path}", flush=True)

    def forward(self, images, views, bags, batch_size):
        features = self.encoder(images) + self.view_embedding(views)
        image_logits = self.classifier(self.dropout(features))
        mask = bags.unsqueeze(0) == torch.arange(batch_size, device=bags.device).unsqueeze(1)
        if self.pooling == "noisy_or":
            # Independent noisy-OR for every disease; no softmax over diseases.
            image_probabilities = image_logits.float().sigmoid().masked_fill(~mask.unsqueeze(-1), 0.0)
            study_probabilities = 1.0 - (1.0-image_probabilities).prod(dim=1)
            study_logits = torch.logit(study_probabilities.clamp(1e-5, 1-1e-5))
        else:
            # Backward-compatible attention pooling for the first completed runs.
            # The softmax is over views only and is independently repeated per disease.
            scores = self.attention(features).unsqueeze(0).expand(batch_size, -1, -1)
            view_attention = scores.float().masked_fill(~mask.unsqueeze(-1), float("-inf")).softmax(dim=1)
            study_logits = (view_attention * image_logits.float().unsqueeze(0)).sum(dim=1)
        return study_logits, image_logits
