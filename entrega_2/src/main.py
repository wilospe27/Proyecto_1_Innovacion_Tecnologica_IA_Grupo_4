"""Punto de entrada del proyecto (interfaz de línea de comandos).

Ejecutar desde la carpeta ``entrega_2/``::

    python -m src.main --help
    python -m src.main especies   --region valle_del_cauca --filtrar-localidad "Valle del Cauca"
    python -m src.main download   --lista data/especies_objetivo.txt --max-per-species 60
    python -m src.main depurar
    python -m src.main preprocess --manifest data/metadata/depuracion.csv --max-seconds 60
    python -m src.main baselines
    python -m src.main train      --model cnn_baseline --epochs 30
    python -m src.main evaluate   --checkpoint experiments/checkpoints/<corrida>.pt --split test
    python -m src.main predict    --audio grabacion.wav --checkpoint experiments/checkpoints/<corrida>.pt

Las importaciones pesadas (PyTorch, librosa) se hacen dentro de cada comando
para que ``--help`` responda rápido.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.utils.helpers import DATA_DIR, METADATA_DIR, PROCESSED_DIR, RAW_DIR, RESULTS_DIR


def read_species_list(path: str | Path) -> list[str]:
    """Lee una especie por línea (ignora líneas vacías y comentarios con #)."""
    names = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            names.append(" ".join(line.replace("_", " ").split()))
    if not names:
        raise ValueError(f"La lista {path} no contiene especies")
    return names


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.main",
        description="Identificación de especies de fauna colombiana por su canto (Grupo 4).",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    species = commands.add_parser(
        "especies",
        help="Rankea las especies de una región con metadatos de xeno-canto (sin descargar audio).",
    )
    species.add_argument("--region", default="valle_del_cauca", help="Región predefinida (valle_del_cauca).")
    species.add_argument(
        "--box", nargs=4, type=float, metavar=("LAT_MIN", "LON_MIN", "LAT_MAX", "LON_MAX"),
        default=None, help="Rectángulo propio; reemplaza a --region.",
    )
    species.add_argument("--pais", default="Colombia")
    species.add_argument("--grupo", default="birds", help="birds, frogs, grasshoppers, bats...")
    species.add_argument("--calidades", nargs="+", default=["A", "B"])
    species.add_argument("--min-grabaciones", type=int, default=30)
    species.add_argument(
        "--filtrar-localidad", default=None,
        help='Deja solo grabaciones cuya localidad contenga este texto, p. ej. "Valle del Cauca".',
    )
    species.add_argument("--max-paginas", type=int, default=None, help="Limita la consulta (pruebas).")
    species.add_argument("--metadatos", default=str(METADATA_DIR / "xeno_canto_region.csv"))
    species.add_argument(
        "--salida", default=str(RESULTS_DIR / "ranking_especies.csv"),
        help="CSV del ranking (se versiona: sustenta la selección de especies en el reporte 2).",
    )
    species.add_argument("--top", type=int, default=40, help="Filas a mostrar en pantalla.")

    download = commands.add_parser(
        "download", help="Descarga metadatos y audios de xeno-canto (requiere la clave XC_API_KEY)."
    )
    which = download.add_mutually_exclusive_group(required=True)
    which.add_argument("--species", nargs="+", help='Nombres científicos, p. ej. "Turdus fuscater".')
    which.add_argument("--lista", help="Archivo de texto con una especie por línea.")
    download.add_argument("--country", default="Colombia", help="País (etiqueta cnt:).")
    download.add_argument("--group", default="birds", help="Grupo de xeno-canto: birds, frogs, grasshoppers, bats...")
    download.add_argument("--calidades", nargs="+", default=["A", "B"], help="Calidades a descargar.")
    download.add_argument("--max-per-species", type=int, default=60)
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
    preprocess.add_argument(
        "--max-seconds", type=float, default=None,
        help="Usa solo los primeros segundos de cada grabación (limita el peso de las grabaciones largas).",
    )
    preprocess.add_argument(
        "--manifest", default=None,
        help="Manifiesto de la depuración; si se indica, solo procesa los archivos conservados.",
    )

    clean = commands.add_parser(
        "depurar", help="Revisa los audios descargados y genera el manifiesto de depuración (no borra nada)."
    )
    clean.add_argument("--raw-dir", default=str(RAW_DIR))
    clean.add_argument("--metadata", default=str(METADATA_DIR / "xeno_canto.csv"))
    clean.add_argument("--lista", default=str(DATA_DIR / "especies_objetivo.txt"))
    clean.add_argument("--min-seconds", type=float, default=1.0)
    clean.add_argument("--manifest", default=str(METADATA_DIR / "depuracion.csv"))
    clean.add_argument(
        "--resumen", default=str(RESULTS_DIR / "depuracion_resumen.csv"),
        help="Resumen por especie (se versiona: sustenta la sección II.C del reporte 2).",
    )

    eda = commands.add_parser("eda", help="Genera las tablas y figuras del análisis exploratorio.")
    eda.add_argument("--metadata", default=str(METADATA_DIR / "xeno_canto.csv"))
    eda.add_argument("--manifest", default=str(METADATA_DIR / "depuracion.csv"))
    eda.add_argument("--processed-dir", default=str(PROCESSED_DIR))

    baselines = commands.add_parser(
        "baselines", help="Evalúa los modelos clásicos de referencia sobre estadísticas de MFCC."
    )
    baselines.add_argument("--processed-dir", default=str(PROCESSED_DIR))
    baselines.add_argument("--split-column", default="split", choices=["split", "split_ventana"])
    baselines.add_argument("--seed", type=int, default=42)

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
    train.add_argument(
        "--split-column", default="split", choices=["split", "split_ventana"],
        help="split = por grabación (correcta); split_ventana = al azar por ventana (solo para H3).",
    )
    train.add_argument("--threads", type=int, default=None, help="Hilos de CPU para PyTorch.")
    train.add_argument(
        "--base-channels", type=int, default=32,
        help="Canales del primer bloque de la CNN base (16 = versión liviana para entrenar en CPU).",
    )

    evaluate = commands.add_parser("evaluate", help="Evalúa un checkpoint en una partición.")
    evaluate.add_argument("--checkpoint", required=True)
    evaluate.add_argument("--processed-dir", default=str(PROCESSED_DIR))
    evaluate.add_argument("--split", default="test", choices=["train", "val", "test"])
    evaluate.add_argument(
        "--split-column", default=None, help="Columna de partición (por defecto, la del entrenamiento)."
    )
    evaluate.add_argument("--batch-size", type=int, default=64)

    predict = commands.add_parser("predict", help="Predice la especie de un archivo de audio.")
    predict.add_argument("--audio", required=True)
    predict.add_argument("--checkpoint", required=True)
    predict.add_argument("--top-k", type=int, default=5)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.command == "especies":
        from src.data.download import species_ranking

        ranking = species_ranking(
            region=None if args.box else args.region,
            box=tuple(args.box) if args.box else None,
            country=args.pais,
            group=args.grupo,
            qualities=args.calidades,
            min_recordings=args.min_grabaciones,
            loc_contains=args.filtrar_localidad,
            metadata_csv=args.metadatos,
            ranking_csv=args.salida,
            max_pages=args.max_paginas,
        )
        print(ranking.head(args.top).to_string(index=False))
        print(f"\nRanking completo: {args.salida}")

    elif args.command == "download":
        from src.data.download import download_species

        download_species(
            args.species or read_species_list(args.lista),
            raw_dir=args.raw_dir,
            metadata_csv=args.metadata,
            country=args.country,
            group=args.group,
            qualities=args.calidades,
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
            max_seconds=args.max_seconds,
        )
        preprocess_directory(args.raw_dir, args.out_dir, cfg, manifest_csv=args.manifest)

    elif args.command == "depurar":
        from src.data.clean import clean_dataset, summarize_manifest

        manifest = clean_dataset(
            args.raw_dir,
            metadata_csv=args.metadata,
            manifest_csv=args.manifest,
            summary_csv=args.resumen,
            target_species=read_species_list(args.lista),
            min_seconds=args.min_seconds,
        )
        print(summarize_manifest(manifest).to_string(index=False))
        print(f"\nManifiesto: {args.manifest}")

    elif args.command == "eda":
        from src.analysis.eda import run_eda

        result = run_eda(args.metadata, args.manifest, args.processed_dir)
        print(json.dumps(result["general"], indent=2, ensure_ascii=False))

    elif args.command == "baselines":
        from src.models.classical import run_baselines

        metrics = run_baselines(args.processed_dir, split_column=args.split_column, seed=args.seed)
        columns = ["modelo", "conjunto", "accuracy", "balanced_accuracy", "f1_macro", "top3_accuracy"]
        print(metrics[columns].round(3).to_string(index=False))

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
                split_column=args.split_column,
                torch_threads=args.threads,
                base_channels=args.base_channels,
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
            split_column=args.split_column,
        )
        print(json.dumps(metrics, indent=2))

    elif args.command == "predict":
        from src.inference.predict import predict_file

        print(predict_file(args.audio, args.checkpoint, top_k=args.top_k).to_string(index=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
