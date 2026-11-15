"""Modelos de clasificación de especies sobre espectrogramas Mel.

- ``cnn_baseline``: CNN compacta entrenada desde cero (modelo de referencia).
- ``efficientnet_b0``: transferencia de aprendizaje con EfficientNet-B0
  preentrenada en ImageNet (``torchvision``).

Ambos reciben tensores ``(lote, 1, n_mels, n_frames)`` y retornan *logits*
``(lote, n_clases)``. Gracias al *pooling* adaptativo aceptan ventanas de
distinta duración.
"""

from __future__ import annotations

from torch import Tensor, nn


class ConvBlock(nn.Module):
    """Dos convoluciones 3x3 con BatchNorm y ReLU, seguidas de MaxPool y Dropout."""

    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout2d(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


class BirdCNN(nn.Module):
    """CNN de referencia: ``n_blocks`` bloques convolucionales + clasificador lineal."""

    def __init__(
        self,
        n_classes: int,
        in_channels: int = 1,
        base_channels: int = 32,
        n_blocks: int = 4,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        if n_classes < 2:
            raise ValueError("n_classes debe ser mayor o igual a 2")
        channels = [in_channels] + [base_channels * 2**i for i in range(n_blocks)]
        self.features = nn.Sequential(
            *[ConvBlock(channels[i], channels[i + 1]) for i in range(n_blocks)]
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(channels[-1], n_classes),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.classifier(self.pool(self.features(x)))


class EfficientNetSpectrogram(nn.Module):
    """EfficientNet-B0 adaptada a espectrogramas de un canal."""

    def __init__(self, n_classes: int, pretrained: bool = True, dropout: float = 0.3) -> None:
        super().__init__()
        from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

        weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = efficientnet_b0(weights=weights)
        in_features = backbone.classifier[1].in_features
        backbone.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_features, n_classes))
        self.backbone = backbone

    def forward(self, x: Tensor) -> Tensor:
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)  # replica el canal para reutilizar los pesos de ImageNet
        return self.backbone(x)


MODEL_REGISTRY: dict[str, type[nn.Module]] = {
    "cnn_baseline": BirdCNN,
    "efficientnet_b0": EfficientNetSpectrogram,
}


def build_model(name: str, n_classes: int, **kwargs) -> nn.Module:
    """Construye un modelo registrado en ``MODEL_REGISTRY``."""
    try:
        model_class = MODEL_REGISTRY[name]
    except KeyError:
        options = ", ".join(sorted(MODEL_REGISTRY))
        raise ValueError(f"Modelo desconocido '{name}'. Opciones: {options}") from None
    return model_class(n_classes=n_classes, **kwargs)


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """Número de parámetros (entrenables por defecto)."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad or not trainable_only)
