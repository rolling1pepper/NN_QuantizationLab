from __future__ import annotations

import torch
from torch import nn


class TinyVisionNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Linear(8, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.features(x)
        flattened = torch.flatten(features, 1)
        return self.classifier(flattened)


class KeywordVisionNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = TinyVisionNet()

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.backbone(pixel_values)


class TinyTextNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(128, 16)
        self.encoder = nn.Linear(16, 16)
        self.classifier = nn.Linear(16, 4)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        del token_type_ids
        embeddings = self.embedding(input_ids)
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (embeddings * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        encoded = torch.relu(self.encoder(pooled))
        logits = self.classifier(encoded)
        return {
            "last_hidden_state": logits.unsqueeze(1),
            "pooler_output": logits,
        }


class TwoInputTextNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(128, 16)
        self.classifier = nn.Linear(16, 4)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        embeddings = self.embedding(input_ids)
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (embeddings * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        return self.classifier(pooled)


class ThirdPartyVisionNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(6),
            nn.GELU(),
            nn.Conv2d(6, 10, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.head = nn.Linear(10, 5)
        self.aux_head = nn.Linear(10, 3)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.stem(x)
        flattened = torch.flatten(features, 1)
        return self.head(flattened), self.aux_head(flattened)


class ThirdPartyTextNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(256, 12)
        self.proj = nn.Linear(12, 12)
        self.head = nn.Linear(12, 6)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor,
    ) -> torch.Tensor:
        del token_type_ids
        embeddings = self.embedding(input_ids)
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (embeddings * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        projected = torch.relu(self.proj(pooled))
        return self.head(projected)


class RuleBasedVisionClassifier(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        signal = x.mean(dim=(1, 2, 3))
        return torch.stack((1.0 - signal, signal), dim=-1)


class RuleBasedTextClassifier(nn.Module):
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor,
    ) -> torch.Tensor:
        del attention_mask, token_type_ids
        signal = input_ids[:, 0].float().remainder(2)
        return torch.stack((1.0 - signal, signal), dim=-1)


def create_factory_vision_model(
    use_pretrained: bool = False,
    num_classes: int = 4,
) -> nn.Module:
    del use_pretrained, num_classes
    return TinyVisionNet().eval()


def create_factory_text_model(
    use_pretrained: bool = False,
    vocab_size: int = 128,
) -> nn.Module:
    del use_pretrained, vocab_size
    return TinyTextNet().eval()


def create_rule_based_vision_model(use_pretrained: bool = False) -> nn.Module:
    del use_pretrained
    return RuleBasedVisionClassifier().eval()


def create_rule_based_text_model(use_pretrained: bool = False) -> nn.Module:
    del use_pretrained
    return RuleBasedTextClassifier().eval()
