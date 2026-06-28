from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def _load(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _best_by(df: pd.DataFrame, metric: str, group_cols: list[str]) -> pd.DataFrame:
    rows = []
    for _, group in df.groupby(group_cols, sort=True):
        rows.append(group.loc[group[metric].idxmax()])
    return pd.DataFrame(rows)


def _metric_policy(dataset: str, label_ratio: float, row: pd.Series) -> str:
    if dataset == "road":
        if label_ratio <= 0.011:
            return "Use CMF-CAN as the label-efficiency result; report F1, AUPR, AUROC, and low-FPR recall together."
        return "Use Transformer for default F1 if needed, but CMF-CAN for ranking/low-FPR claims."
    if dataset == "ctt_test01":
        if label_ratio < 1.0:
            return "Report few-label stability with 3 seeds; avoid overclaiming CMF-CAN when Transformer/concat wins F1."
        return "Use calibrated low-FPR threshold; raw default F1 understates same-domain performance."
    return "Use low-FPR and AUPR as primary deployment metrics."


def build(root: Path) -> tuple[pd.DataFrame, str]:
    table_dir = root / "results/cmf_tables"
    road = _load(table_dir / "road_few_label_3seed_mean_std.csv")
    ctt = _load(table_dir / "ctt_few_label_3seed_mean_std.csv")

    rows = []
    if not road.empty:
        road = road.copy()
        road["dataset"] = "road"
        for _, row in road.iterrows():
            rows.append(row.to_dict())
    if not ctt.empty:
        for _, row in ctt.iterrows():
            rows.append(row.to_dict())

    all_rows = pd.DataFrame(rows)
    if all_rows.empty:
        raise FileNotFoundError("missing few-label summary tables")

    group_cols = ["dataset", "label_ratio"]
    best_f1 = _best_by(all_rows, "f1_mean", group_cols)
    best_aupr = _best_by(all_rows, "aupr_mean", group_cols)
    best_lowfpr = _best_by(all_rows, "recall_at_fpr_1em04_mean", group_cols)

    out_rows = []
    for key, group in all_rows.groupby(group_cols, sort=True):
        dataset, label_ratio = key
        f1 = best_f1[(best_f1["dataset"] == dataset) & (best_f1["label_ratio"] == label_ratio)].iloc[0]
        aupr = best_aupr[(best_aupr["dataset"] == dataset) & (best_aupr["label_ratio"] == label_ratio)].iloc[0]
        low = best_lowfpr[(best_lowfpr["dataset"] == dataset) & (best_lowfpr["label_ratio"] == label_ratio)].iloc[0]
        cmf = group[group["model"] == "cmf_can"]
        cmf_row = cmf.iloc[0] if len(cmf) else group.iloc[0]
        out_rows.append(
            {
                "dataset": dataset,
                "label_ratio": float(label_ratio),
                "models_tested": ",".join(sorted(group["model"].astype(str).unique())),
                "best_f1_model": f1["model"],
                "best_f1_mean": f1["f1_mean"],
                "best_f1_std": f1["f1_std"],
                "best_aupr_model": aupr["model"],
                "best_aupr_mean": aupr["aupr_mean"],
                "best_lowfpr_model": low["model"],
                "best_recall_at_fpr_1em04_mean": low["recall_at_fpr_1em04_mean"],
                "cmf_can_f1_mean": cmf_row["f1_mean"],
                "cmf_can_aupr_mean": cmf_row["aupr_mean"],
                "cmf_can_recall_at_fpr_1em04_mean": cmf_row["recall_at_fpr_1em04_mean"],
                "recommended_metric_policy": _metric_policy(dataset, float(label_ratio), cmf_row),
            }
        )

    out = pd.DataFrame(out_rows).sort_values(["dataset", "label_ratio"])
    out.to_csv(table_dir / "label_ratio_coverage_summary.csv", index=False)

    lines = ["# Label Ratio Coverage and Metric Policy", "", "Date: 2026-06-27", ""]
    lines.append("## Coverage")
    lines.append("")
    lines.append("| Dataset | Ratios covered | Models |")
    lines.append("|---|---|---|")
    for dataset, group in out.groupby("dataset", sort=True):
        ratios = ", ".join(f"{r:g}" for r in sorted(group["label_ratio"].unique()))
        models = "; ".join(sorted(set(",".join(group["models_tested"]).split(","))))
        lines.append(f"| {dataset} | {ratios} | {models} |")
    lines.append("")
    lines.append("## Best Result by Ratio")
    lines.append("")
    lines.append("| Dataset | Ratio | Best F1 model | F1 | Best AUPR model | AUPR | Best low-FPR model | Recall@FPR<=1e-4 | CMF-CAN F1 | CMF-CAN AUPR | Policy |")
    lines.append("|---|---:|---|---:|---|---:|---|---:|---:|---:|---|")
    for rec in out.to_dict("records"):
        lines.append(
            f"| {rec['dataset']} | {rec['label_ratio']:g} | {rec['best_f1_model']} | {rec['best_f1_mean']:.4f} | "
            f"{rec['best_aupr_model']} | {rec['best_aupr_mean']:.4f} | {rec['best_lowfpr_model']} | "
            f"{rec['best_recall_at_fpr_1em04_mean']:.4f} | {rec['cmf_can_f1_mean']:.4f} | "
            f"{rec['cmf_can_aupr_mean']:.4f} | {rec['recommended_metric_policy']} |"
        )
    lines.append("")
    lines.append("## Metric Repair Decision")
    lines.append("")
    lines.append("- Do not replace metrics merely because a result is poor; keep default F1 for comparability.")
    lines.append("- For imbalanced and deployment-oriented IDS, promote AUPR, AUROC, Recall@FPR<=1e-4/5e-4/1e-3, ECE, and calibrated-threshold F1 as primary evidence.")
    lines.append("- For CT&T shifted-domain splits, default F1 is a threshold-transfer diagnostic, not the only performance target.")
    lines.append("- For ROAD few-label, CMF-CAN should be claimed as label-efficient/ranking/low-FPR, while Transformer can still be acknowledged as stronger on default F1 at several higher ratios.")
    report = "\n".join(lines) + "\n"
    (table_dir / "LABEL_RATIO_COVERAGE_REPORT.md").write_text(report, encoding="utf-8")
    return out, report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    args = parser.parse_args()
    out, _ = build(Path(args.root).resolve())
    print(f"[label_ratio_report] wrote results/cmf_tables/label_ratio_coverage_summary.csv rows={len(out)}")
    print("[label_ratio_report] wrote results/cmf_tables/LABEL_RATIO_COVERAGE_REPORT.md")


if __name__ == "__main__":
    main()
