"""Punto de entrada del proyecto (interfaz de línea de comandos).

Ejecutar desde la carpeta ``entrega_1/``::

    python -m src.main --help
    python -m src.main download   --species "Turdus fuscater" "Pitangus sulphuratus" --max-per-species 30
    python -m src.main preprocess --raw-dir data/raw --out-dir data/processed
    python -m src.main train      --model cnn_baseline --epochs 30
    python -m src.main evaluate   --checkpoint experiments/checkpoints/<corrida>.pt --split test
    python -m src.main predict    --audio grabacion.wav --checkpoint experiments/checkpoints/<corrida>.pt

Las importaciones pesadas (PyTorch, librosa) se hacen dentro de cada comando
para que ``--help`` responda rápido.
"""

from __future__ import annotations

import argparse
import json

from src.utils.helpers import METADATA_DIR, PROCESSED_DIR, RAW_DIR


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.main",
        description="Identificación de especies de fauna colombiana por su canto (Grupo 4).",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    download = commands.add_parser(
        "download", help="Descarga metadatos y audios de xeno-canto (requiere XC_API_KEY)."
    )
    download.add_argument(
        "--species", nargs="+", required=True, help='Nombres científicos, p. ej. "Turdus fuscater".'
    )
    download.add_argument("--country", default="Colombia", help="País (etiqueta cnt:).")
    download.add_argument(
        "--group", default="birds", help="Grupo de xeno-canto: birds, frogs, grasshoppers, bats..."
    )
    download.add_argument("--quality", default=None, help="Calidad exacta de la grabación (A-E).")
    download.add_argument("--max-per-species", type=int, default=30)
    download.add_argument("--raw-dir", default=str(RAW_DIR))
    download.add_argument("--metadata", default=str(METADATA_DIR / "xeno_canto.csv"))

    preprocess = commands.add_parser(
        "preprocess", help="Segmenta los audios y genera espectrogramas Mel (.npy) + índice."
    )
    preprocess.add_argument("--raw-dir", default=str(RAW_DIR))
    preprocess.add_argument("--out-dir", default=str(PROCESSED_DIR))
    preprocess.add_argument("--sample-rate", type=int, default=32_000)
    preprocess.add_argument("--segment-seconds", type=float, default=5.0)
    preprocess.add_argument("--hop-seconds", type=float, default=2.5)
    preprocess.add_argument("--n-mels", type=int, default=128)
    preprocess.add_argument("--reduce-noise", action="store_true", help="Aplica noisereduce.")

    train = commands.add_parser("train", help="Entrena un modelo sobre las ventanas procesadas.")
    train.add_argument("--processed-dir", default=str(PROCESSED_DIR))
    train.add_argument("--model", default="cnn_baseline", choices=["cnn_baseline", "efficientnet_b0"])
    train.add_argument("--epochs", type=int, default=30)
    train.add_argument("--batch-size", type=int, default=32)
    train.add_argument("--lr", type=float, default=1e-3)
    train.add_argument("--patience", type=int, default=5)
    train.add_argument("--no-augment", action="store_true", help="Desactiva SpecAugment.")
    train.add_argument("--no-class-weights", action="store_true", help="Sin pesos por clase.")
    train.add_argument("--no-pretrained", action="store_true", help="EfficientNet sin ImageNet.")
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--run-name", default=None)

    evaluate = commands.add_parser("evaluate", help="Evalúa un checkpoint en una partición.")
    evaluate.add_argument("--checkpoint", required=True)
    evaluate.add_argument("--processed-dir", default=str(PROCESSED_DIR))
    evaluate.add_argument("--split", default="test", choices=["train", "val", "test"])
    evaluate.add_argument("--batch-size", type=int, default=64)

    predict = commands.add_parser("predict", help="Predice la especie de un archivo de audio.")
    predict.add_argument("--audio", required=True)
    predict.add_argument("--checkpoint", required=True)
    predict.add_argument("--top-k", type=int, default=5)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.command == "download":
        from src.data.download import download_species

        download_species(
            args.species,
            raw_dir=args.raw_dir,
            metadata_csv=args.metadata,
            country=args.country,
            group=args.group,
            quality=args.quality,
            max_per_species=args.max_per_species,
        )

    elif args.command == "preprocess":
        from src.data.preprocess import AudioConfig, preprocess_directory

        cfg = AudioConfig(
            sample_rate=args.sample_rate,
            segment_seconds=args.segment_seconds,
            hop_seconds=args.hop_seconds,
            n_mels=args.n_mels,
            reduce_noise=args.reduce_noise,
        )
        preprocess_directory(args.raw_dir, args.out_dir, cfg)

    elif args.command == "train":
        from src.training.train import TrainConfig, train

        summary = train(
            TrainConfig(
                processed_dir=args.processed_dir,
                model_name=args.model,
                pretrained=not args.no_pretrained,
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.lr,
                patience=args.patience,
                augment=not args.no_augment,
                class_weights=not args.no_class_weights,
                seed=args.seed,
                run_name=args.run_name,
            )
        )
        print(json.dumps({k: v for k, v in summary.items() if k != "config"}, indent=2))

    elif args.command == "evaluate":
        from src.evaluation.evaluate import evaluate_checkpoint

        metrics = evaluate_checkpoint(
            args.checkpoint,
            processed_dir=args.processed_dir,
            split=args.split,
            batch_size=args.batch_size,
        )
        print(json.dumps(metrics, indent=2))

    elif args.command == "predict":
        from src.inference.predict import predict_file

        print(predict_file(args.audio, args.checkpoint, top_k=args.top_k).to_string(index=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
