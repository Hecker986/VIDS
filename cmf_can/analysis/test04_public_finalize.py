from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


ROOT = Path(".")
OUT = ROOT / "results/test04_public_reproduction"
TABLES = OUT / "tables"
FIGS = OUT / "figures"
PREDS = OUT / "predictions"
FPR_BUDGETS = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2]


def setup() -> None:
    for p in [TABLES, FIGS, OUT / "audits", OUT / "manifests"]:
        p.mkdir(parents=True, exist_ok=True)
    mpl.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.linewidth": 0.9,
            "svg.fonttype": "none",
        }
    )


def write_table(df: pd.DataFrame, name: str) -> None:
    df.to_csv(TABLES / f"{name}.csv", index=False)
    (TABLES / f"{name}.tex").write_text(
        df.to_latex(index=False, escape=True, na_rep="NA", float_format=lambda x: f"{x:.4f}"),
        encoding="utf-8",
    )


def recall_at_budget(y: np.ndarray, score: np.ndarray, budget: float) -> dict:
    order = np.argsort(-score)
    ys = y[order]
    ss = score[order]
    pos = max(int((y == 1).sum()), 1)
    neg = max(int((y == 0).sum()), 1)
    tp = fp = 0
    best = {"recall": 0.0, "precision": 1.0, "f1": 0.0, "actual_fpr": 0.0, "threshold": np.inf}
    for label, threshold in zip(ys, ss):
        if int(label) == 1:
            tp += 1
        else:
            fp += 1
        fpr = fp / neg
        if fpr <= budget:
            rec = tp / pos
            prec = tp / max(tp + fp, 1)
            best = {
                "recall": rec,
                "precision": prec,
                "f1": 2 * prec * rec / max(prec + rec, 1e-12),
                "actual_fpr": fpr,
                "threshold": float(threshold),
            }
        else:
            break
    return best


def key_score_files(public: pd.DataFrame) -> list[Path]:
    completed = public[(public["setting"].eq("ctt_test04")) & (public["status"].eq("completed"))].copy()
    picks = []
    selectors = [
        ("SAFE_CAN", "HistGradientBoosting", "C_5x_negative_cap"),
        ("P1_public_default", "HistGradientBoosting", "C_5x_negative_cap"),
        ("RISKY_PROTOCOL", "HistGradientBoosting", "C_5x_negative_cap"),
        ("P2_no_timestamp", "HistGradientBoosting", "C_5x_negative_cap"),
        ("SAFE_CAN", "GradientBoosting", "C_5x_negative_cap"),
    ]
    for protocol, model, neg in selectors:
        p = PREDS / f"test04_{protocol}_{model}_{neg}_seed42_scores.csv"
        if p.exists():
            picks.append(p)
    if not picks and not completed.empty:
        best = completed.sort_values("f1", ascending=False).iloc[0]
        p = PREDS / f"test04_{best.feature_protocol}_{best.model}_{best.negative_protocol}_seed{int(best.seed)}_scores.csv"
        if p.exists():
            picks.append(p)
    return picks


def low_fpr_and_event(public: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for path in key_score_files(public):
        print(f"[test04_finalize] read score {path}", flush=True)
        df = pd.read_csv(path, usecols=["feature_protocol", "model", "label", "score", "prediction_val_f1_threshold"])
        y = df["label"].astype(np.int8).to_numpy()
        score = df["score"].astype(np.float32).to_numpy()
        protocol = str(df["feature_protocol"].iloc[0])
        model = str(df["model"].iloc[0])
        for budget in FPR_BUDGETS:
            b = recall_at_budget(y, score, budget)
            rows.append(
                {
                    "score_file": str(path),
                    "feature_protocol": protocol,
                    "model": model,
                    "setting": "ctt_test04",
                    "threshold_type": "best_test_upper_bound",
                    "fpr_budget": budget,
                    "recall_at_fpr": b["recall"],
                    "precision_at_fpr": b["precision"],
                    "f1_at_fpr": b["f1"],
                    "actual_fpr": b["actual_fpr"],
                    "threshold": b["threshold"],
                    "auroc": roc_auc_score(y, score) if len(np.unique(y)) == 2 else np.nan,
                    "aupr": average_precision_score(y, score) if len(np.unique(y)) == 2 else np.nan,
                    "num_positive": int((y == 1).sum()),
                    "num_negative": int((y == 0).sum()),
                }
            )
        pred = df["prediction_val_f1_threshold"].astype(np.int8).to_numpy()
        starts = np.where((y == 1) & np.r_[True, y[:-1] == 0])[0]
        ends = np.r_[starts[1:], len(y)]
        hit = 0
        for s, e in zip(starts, ends):
            if (pred[s:e] == 1).any():
                hit += 1
        rows.append(
            {
                "score_file": str(path),
                "feature_protocol": protocol,
                "model": model,
                "setting": "ctt_test04",
                "threshold_type": "validation_f1_threshold_event_approx",
                "event_recall": hit / max(len(starts), 1),
                "num_events": int(len(starts)),
                "event_boundary_quality": "approximate_label_transition",
            }
        )
    out = pd.DataFrame(rows)
    write_table(out, "final_low_fpr_event_metrics")
    return out


def plot_bar(df: pd.DataFrame, name: str, title: str, top_n: int = 12) -> None:
    plot = df[(df["setting"].eq("ctt_test04")) & (df["status"].eq("completed"))].copy()
    plot = plot.sort_values("f1", ascending=False).head(top_n)
    fig, ax = plt.subplots(figsize=(7.2, 3.3))
    labels = plot["feature_protocol"].astype(str) + "\n" + plot["model"].astype(str)
    hatches = ["//", "\\\\", "xx", "..", "++", "--"]
    for i, (_, row) in enumerate(plot.iterrows()):
        ax.bar(i, row["f1"], color="#D9D9D9", edgecolor="black", hatch=hatches[i % len(hatches)], linewidth=0.8)
    ax.set_xticks(np.arange(len(plot)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("F1")
    ax.set_ylim(0, 1.05)
    ax.set_title(title)
    ax.grid(axis="y", color="#E6E6E6", linewidth=0.6)
    fig.savefig(FIGS / f"{name}.svg", bbox_inches="tight")
    plt.close(fig)


def plot_low(low: pd.DataFrame) -> None:
    plot = low[low["threshold_type"].eq("best_test_upper_bound")].copy()
    fig, ax = plt.subplots(figsize=(6.2, 3.3))
    for (protocol, model), g in plot.groupby(["feature_protocol", "model"]):
        ax.plot(g["fpr_budget"], g["recall_at_fpr"], marker="o", linewidth=1.1, label=f"{protocol}/{model}"[:34])
    ax.set_xscale("log")
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("FPR budget")
    ax.set_ylabel("Recall")
    ax.grid(axis="y", color="#E6E6E6", linewidth=0.6)
    if not plot.empty:
        ax.legend(frameon=False, fontsize=6)
    fig.savefig(FIGS / "final_low_fpr_event_metrics.svg", bbox_inches="tight")
    fig.savefig(FIGS / "paper_fig6_low_fpr_event.svg", bbox_inches="tight")
    plt.close(fig)


def write_reports(public: pd.DataFrame, low: pd.DataFrame) -> None:
    completed = public[(public["setting"].eq("ctt_test04")) & (public["status"].eq("completed"))].copy()
    best = completed.sort_values("f1", ascending=False)
    safe = best[best["feature_protocol"].isin(["SAFE_CAN", "P2_no_timestamp", "P6_delta_features"])]
    risky = best[best["feature_protocol"].isin(["RISKY_PROTOCOL", "P1_public_default", "P4_timestamp_only", "P7_public_plus_delta"])]
    best_f1 = float(best["f1"].max())
    safe_f1 = float(safe["f1"].max()) if not safe.empty else np.nan
    risky_f1 = float(risky["f1"].max()) if not risky.empty else np.nan
    strategy = "C. Continue method/protocol reproduction"
    if best_f1 >= 0.95 and safe_f1 >= 0.8:
        strategy = "A. Safe-CAN GRAIN-CAN method paper"
    elif risky_f1 > safe_f1 + 0.2:
        strategy = "B. Shortcut-aware benchmark correction / measurement paper"

    (OUT / "public_protocol_reproduction.md").write_text(
        "# Public Protocol Reproduction\n\n"
        f"Best local CT&T test04 F1: {best_f1:.4f}. Public-style P1 did not reproduce a 0.998-level result on the local `set_01` data.\n\n"
        f"```csv\n{best[['feature_protocol','model','negative_protocol','seed','f1','auroc','aupr','fpr','fnr']].head(25).to_csv(index=False)}```\n",
        encoding="utf-8",
    )
    (OUT / "timestamp_shortcut_audit.md").write_text(
        "# Timestamp Shortcut Audit\n\n"
        f"Best Safe-CAN-family F1: {safe_f1:.4f}. Best risky/public-family F1: {risky_f1:.4f}. Timestamp/public/risky features did not reproduce the reported 0.998 result on the local data.\n",
        encoding="utf-8",
    )
    (OUT / "timestamp_ablation.md").write_text(
        "# Timestamp Ablation\n\nRaw timestamp did not explain the public high result on the local `set_01`: P1_public_default and P4_timestamp_only remain far below the Safe-CAN/delta-feature best rows.\n",
        encoding="utf-8",
    )
    (OUT / "safe_vs_risky_features.md").write_text(
        "# Safe vs Risky Features\n\n"
        f"Safe-CAN best F1: {safe_f1:.4f}. Risky/public best F1: {risky_f1:.4f}. The best result is Safe-CAN/HGB with 5x negatives, not a raw timestamp shortcut, but it is still far below 0.998.\n",
        encoding="utf-8",
    )
    (OUT / "shortcut_finding_report.md").write_text(
        "# Shortcut Finding Report\n\n"
        "The local experiment did not reproduce the public 0.998 test04 result with public default features, raw timestamp only, risky protocol features, no-timestamp public features, or Safe-CAN features. The most likely explanations are data-version mismatch, unreported preprocessing/label protocol, or a different train/test construction in the public result.\n\n"
        f"Best rows:\n\n```csv\n{best[['feature_protocol','model','negative_protocol','seed','f1','auroc','aupr','fpr','fnr']].head(20).to_csv(index=False)}```\n",
        encoding="utf-8",
    )
    (OUT / "final_low_fpr_event_metrics.md").write_text(
        "# Final Low-FPR And Event Metrics\n\nLow-FPR rows are computed from selected key test04 score dumps and marked as best-test upper bounds. Event rows use approximate label-transition events.\n\n"
        f"```csv\n{low.head(60).to_csv(index=False)}```\n",
        encoding="utf-8",
    )
    (OUT / "final_security4_strategy.md").write_text(
        "# Final Security4 Strategy\n\n"
        f"**Selected direction: {strategy}.**\n\n"
        f"- Best local public-reproduction test04 F1: {best_f1:.4f}.\n"
        f"- Best Safe-CAN F1: {safe_f1:.4f}.\n"
        f"- Best risky/public F1: {risky_f1:.4f}.\n\n"
        "Conclusion: the public 0.998 test04 result is not reproduced on the current local set_01 with the tested feature protocols. To keep a CCF A/Security Four path, the next evidence must obtain the exact public v1.5 data/protocol or turn this into a rigorous benchmark-protocol discrepancy paper.\n",
        encoding="utf-8",
    )
    (OUT / "recommended_next_paper_claims.md").write_text(
        "# Recommended Next Paper Claims\n\n"
        "- Claim that public-style CT&T test04 high results were not reproduced on local set_01 under tested protocols.\n"
        "- Claim Safe-CAN delta/payload timing features are stronger than raw timestamp/public default in this local protocol.\n"
        "- Claim benchmark-protocol discrepancy as a hypothesis, not as proven shortcut, until exact v1.5/public preprocessing is obtained.\n",
        encoding="utf-8",
    )
    (OUT / "unsafe_claims_do_not_write.md").write_text(
        "# Unsafe Claims Do Not Write\n\n"
        "- Do not claim test04 0.998 was reproduced.\n"
        "- Do not claim timestamp shortcut is proven as the source of public 0.998; local timestamp/risky rows did not reproduce it.\n"
        "- Do not claim Safe-CAN reaches Security-Four method-level performance; best local F1 is below 0.8.\n"
        "- Do not claim local set_01 is equivalent to can-train-and-test-v1.5.\n"
        "- Do not use best-test low-FPR thresholds as formal validation-threshold deployment evidence.\n",
        encoding="utf-8",
    )


def main() -> None:
    setup()
    public = pd.read_csv(TABLES / "public_protocol_reproduction.csv")
    timestamp = public[(public["setting"].eq("ctt_test04")) & (public["feature_protocol"].isin(["P1_public_default", "P2_no_timestamp", "P4_timestamp_only", "RISKY_PROTOCOL", "SAFE_CAN", "P7_public_plus_delta"]))].copy()
    ablation = public[(public["setting"].eq("ctt_test04")) & (public["feature_protocol"].isin(["P1_public_default", "P2_no_timestamp", "P4_timestamp_only", "P6_delta_features", "P7_public_plus_delta"]))].copy()
    safe = public[(public["setting"].eq("ctt_test04")) & (public["feature_protocol"].isin(["SAFE_CAN", "RISKY_PROTOCOL", "P1_public_default", "P2_no_timestamp", "P7_public_plus_delta"]))].copy()
    write_table(timestamp, "timestamp_shortcut_audit")
    write_table(ablation, "timestamp_ablation")
    write_table(safe, "safe_vs_risky_features")
    low = low_fpr_and_event(public)
    plot_bar(public, "public_protocol_reproduction", "Public Protocol Test04 Reproduction")
    plot_bar(public, "paper_fig1_public_reproduction", "Public Protocol Test04 Reproduction")
    plot_bar(public, "paper_fig2_feature_protocols", "Feature Protocol Comparison")
    plot_bar(timestamp, "timestamp_shortcut_audit", "Timestamp Shortcut Audit")
    plot_bar(timestamp, "paper_fig3_timestamp_shortcut", "Timestamp Shortcut Audit")
    plot_bar(safe, "safe_vs_risky_features", "Safe vs Risky Features")
    plot_bar(safe, "paper_fig4_safe_vs_risky", "Safe vs Risky Features")
    plot_bar(public, "paper_fig5_test04_leaderboard", "Test04 Leaderboard")
    plot_low(low)
    write_reports(public, low)
    (OUT / "inventory.txt").write_text("\n".join(str(p) for p in sorted(OUT.rglob("*")) if p.is_file()) + "\n", encoding="utf-8")
    print("[test04_public_finalize] done")


if __name__ == "__main__":
    main()
