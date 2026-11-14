"""Pruebas de las arquitecturas (formas de entrada/salida y fábrica de modelos)."""

import pytest
import torch

from src.models.cnn_baseline import BirdCNN, build_model, count_parameters


def test_salida_de_la_cnn_base():
    model = BirdCNN(n_classes=5).eval()
    with torch.no_grad():
        logits = model(torch.randn(2, 1, 128, 501))  # ventana de 5 s con la configuración por defecto
    assert logits.shape == (2, 5)


def test_cnn_acepta_ventanas_de_otro_tamano():
    model = build_model("cnn_baseline", n_classes=3).eval()
    with torch.no_grad():
        assert model(torch.randn(3, 1, 64, 200)).shape == (3, 3)


def test_conteo_de_parametros():
    assert count_parameters(BirdCNN(n_classes=10)) > 0


def test_numero_de_clases_invalido():
    with pytest.raises(ValueError):
        BirdCNN(n_classes=1)


def test_modelo_desconocido():
    with pytest.raises(ValueError, match="Modelo desconocido"):
        build_model("no_existe", n_classes=3)


def test_efficientnet_sin_pesos_preentrenados():
    pytest.importorskip("torchvision")
    model = build_model("efficientnet_b0", n_classes=4, pretrained=False).eval()
    with torch.no_grad():
        assert model(torch.randn(2, 1, 128, 256)).shape == (2, 4)
