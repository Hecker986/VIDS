from __future__ import annotations

import argparse
from pathlib import Path

from cmf_can.training.train import run_training


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--label-ratio", type=float, default=1.0)
    parser.add_argument("--table", default="")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--aux-loss-weight", type=float, default=0.2)
    parser.add_argument("--loss", choices=["ce", "focal"], default="ce")
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--focal-alpha", type=float, default=None)
    parser.add_argument("--sampler", choices=["none", "weighted"], default="none")
    parser.add_argument("--class-weights", choices=["none", "balanced"], default="balanced")
    parser.add_argument("--supcon-weight", type=float, default=0.0)
    parser.add_argument("--supcon-temperature", type=float, default=0.1)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--save-predictions", "--save_predictions", action="store_true")
    parser.add_argument("--save-gate-weights", "--save_gate_weights", action="store_true")
    parser.add_argument(
        "--selection-metric",
        choices=[
            "f1",
            "macro_f1",
            "aupr",
            "recall_at_fpr_1em04",
            "f1_at_fpr_1em04",
            "recall_at_fpr_5em04",
            "f1_at_fpr_5em04",
            "recall_at_fpr_1em03",
            "f1_at_fpr_1em03",
        ],
        default="f1",
    )
    args = parser.parse_args()
    run_training(
        root=Path(args.root).resolve(),
        dataset=args.dataset,
        model_name=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        label_ratio=args.label_ratio,
        table=args.table or None,
        num_workers=args.num_workers,
        aux_loss_weight=args.aux_loss_weight,
        loss_name=args.loss,
        focal_gamma=args.focal_gamma,
        focal_alpha=args.focal_alpha,
        sampler_name=args.sampler,
        selection_metric=args.selection_metric,
        class_weights_name=args.class_weights,
        supcon_weight=args.supcon_weight,
        supcon_temperature=args.supcon_temperature,
        eval_only=args.eval_only,
        checkpoint=args.checkpoint or None,
        save_predictions=args.save_predictions,
        save_gate_weights=args.save_gate_weights,
    )


if __name__ == "__main__":
    main()
