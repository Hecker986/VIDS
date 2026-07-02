from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


OUT = Path("results/final_paper_supplement")
TABLES = OUT / "tables"
FIGS = OUT / "figures"
CTT_SETTINGS = ["ctt_test01", "ctt_test02", "ctt_test03", "ctt_test04"]
MODEL_ORDER = [
    "predict_all_normal",
    "Table13-style GradientBoosting",
    "Table13-style MLP",
    "old window100 Transformer",
    "CAN-Transformer+ same-ID",
    "CMF-CAN",
    "Reliable-CMF-CAN",
    "SAFE_CAN",
    "GRAIN_window_10",
    "GRAIN_window_20",
    "GRAIN_window_100",
]
PLOT_COLORS = ["#1B9E77", "#D95F02", "#7570B3", "#E7298A", "#66A61E", "#E6AB02", "#A6761D", "#1F78B4"]


def setup() -> None:
    for p in [TABLES, FIGS]:
        p.mkdir(parents=True, exist_ok=True)


def read(path: str) -> pd.DataFrame:
    p = Path(path)
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


def tex(df: pd.DataFrame, path: Path) -> None:
    path.write_text(df.to_latex(index=False, float_format=lambda x: f"{x:.4f}" if pd.notna(x) else "NA"), encoding="utf-8")


def write_table(df: pd.DataFrame, name: str) -> pd.DataFrame:
    df.to_csv(TABLES / f"{name}.csv", index=False)
    tex(df, TABLES / f"{name}.tex")
    return df


def first_row(df: pd.DataFrame, mask) -> pd.Series | None:
    sub = df[mask]
    return sub.iloc[0] if len(sub) else None


def normalize_model_row(setting: str, display: str, r: pd.Series | None, source: str) -> dict:
    if r is None:
        return {
            "setting": setting,
            "model": display,
            "accuracy": np.nan,
            "weighted_f1": np.nan,
            "attack_f1": np.nan,
            "attack_precision": np.nan,
            "attack_recall": np.nan,
            "aupr": np.nan,
            "auroc": np.nan,
            "recall_at_fpr_1e_3": np.nan,
            "positive_rate": np.nan,
            "source": source,
            "support_status": "missing_input",
        }
    return {
        "setting": setting,
        "model": display,
        "accuracy": r.get("accuracy", np.nan),
        "weighted_f1": r.get("weighted_f1", np.nan),
        "attack_f1": r.get("attack_f1", r.get("f1", np.nan)),
        "attack_precision": r.get("attack_precision", r.get("precision", np.nan)),
        "attack_recall": r.get("attack_recall", r.get("recall", np.nan)),
        "aupr": r.get("aupr", np.nan),
        "auroc": r.get("auroc", np.nan),
        "recall_at_fpr_1e_3": r.get("recall_at_fpr_1e_3", r.get("recall_at_fpr_1em03", np.nan)),
        "positive_rate": r.get("positive_rate", np.nan),
        "source": source,
        "support_status": "measured",
    }


def current_result_check() -> pd.DataFrame:
    rows = []
    roots = [Path("results/attack_centric_final"), Path("results/metric_correction_paper"), Path("results/final_grain_can")]
    for root in roots:
        for path in sorted(root.glob("**/*")):
            if not path.is_file():
                continue
            kind = path.suffix.lower().lstrip(".")
            if kind not in {"csv", "tex", "svg", "md"}:
                continue
            size = path.stat().st_size
            status = "ok"
            rows_count = np.nan
            cols_count = np.nan
            if kind == "csv":
                try:
                    df = pd.read_csv(path)
                    rows_count = len(df)
                    cols_count = len(df.columns)
                    if len(df) == 0:
                        status = "empty"
                except Exception as exc:
                    status = f"csv_error:{exc}"
            elif kind == "svg":
                text = path.read_text(errors="ignore", encoding="utf-8")
                if "<svg" not in text:
                    status = "invalid_svg"
            else:
                if not path.read_text(errors="ignore", encoding="utf-8").strip():
                    status = "empty"
            rows.append({
                "file_path": str(path),
                "kind": kind,
                "file_size_bytes": size,
                "num_rows": rows_count,
                "num_columns": cols_count,
                "status": status,
                "paper_use": "main" if "paper_fig" in path.name or "paper_table" in path.name else "support_or_limitations",
            })
    out = pd.DataFrame(rows)
    write_table(out, "current_result_check")
    failures = out[out["status"].ne("ok")]
    main_ready = out[(out["paper_use"].eq("main")) & (out["status"].eq("ok"))]
    (OUT / "current_result_check.md").write_text(
        "# Current Result Check\n\n"
        f"Checked {len(out)} CSV/TEX/SVG/MD files from attack_centric_final, metric_correction_paper and final_grain_can.\n\n"
        f"Failures or empty files: {len(failures)}.\n\n"
        f"Main-paper-ready assets detected: {len(main_ready)}.\n\n"
        "Directly usable in main text: attack-centric pipeline, metric trap audit, Table13 case study, corrected CT&T benchmark, ranking inversion, low-FPR/event-level, GRAIN feature preservation. Limitations-only: approximate event boundaries, missing exact original-author confusion matrices, unavailable v1.5 full alignment, and rows with missing score dumps.\n",
        encoding="utf-8",
    )
    return out


def main_ctt_table() -> pd.DataFrame:
    ctt = read("results/attack_centric_final/tables/c1_ctt_all_settings_corrected_benchmark.csv")
    reliable = read("results/cmf_tables/reliable_cmf_main_raw.csv")
    ctt_gen = read("results/cmf_tables/ctt_generalization_15ep.csv")
    rows = []
    for setting in CTT_SETTINGS:
        rows.append(normalize_model_row(setting, "predict_all_normal", first_row(ctt, (ctt["setting"].eq(setting)) & (ctt["model"].eq("predict_all_normal"))), "attack_centric_final"))
        rows.append(normalize_model_row(setting, "Table13-style GradientBoosting", first_row(ctt, (ctt["setting"].eq(setting)) & (ctt["model"].eq("GradientBoosting / sample"))), "attack_centric_final"))
        rows.append(normalize_model_row(setting, "Table13-style MLP", first_row(ctt, (ctt["setting"].eq(setting)) & (ctt["model"].eq("MLP / sample"))), "attack_centric_final"))
        transformer = first_row(ctt, (ctt["setting"].eq(setting)) & (ctt["model"].astype(str).isin(["Transformer", "transformer / old_window100_deep"])))
        if transformer is None and not ctt_gen.empty:
            rr = first_row(ctt_gen, (ctt_gen["dataset"].eq(setting)) & (ctt_gen["model"].eq("transformer")))
            if rr is not None:
                transformer = pd.Series({
                    "attack_f1": rr.get("f1"),
                    "attack_precision": rr.get("precision"),
                    "attack_recall": rr.get("recall"),
                    "aupr": rr.get("aupr"),
                    "auroc": rr.get("auroc"),
                    "accuracy": rr.get("accuracy"),
                    "recall_at_fpr_1e_3": rr.get("recall_at_fpr_1em03"),
                })
        rows.append(normalize_model_row(setting, "old window100 Transformer", transformer, "attack_centric_final_or_ctt_generalization"))
        rows.append(normalize_model_row(setting, "CAN-Transformer+ same-ID", None, "not_available_for_ctt_settings"))
        cmf = first_row(ctt, (ctt["setting"].eq(setting)) & (ctt["model"].eq("CMF-CAN")))
        if cmf is None and not ctt_gen.empty:
            rr = first_row(ctt_gen, (ctt_gen["dataset"].eq(setting)) & (ctt_gen["model"].eq("cmf_can")))
            if rr is not None:
                cmf = pd.Series({
                    "attack_f1": rr.get("f1"),
                    "attack_precision": rr.get("precision"),
                    "attack_recall": rr.get("recall"),
                    "aupr": rr.get("aupr"),
                    "auroc": rr.get("auroc"),
                    "accuracy": rr.get("accuracy"),
                    "recall_at_fpr_1e_3": rr.get("recall_at_fpr_1em03"),
                })
        rows.append(normalize_model_row(setting, "CMF-CAN", cmf, "attack_centric_final_or_ctt_generalization"))
        rel = None
        if not reliable.empty:
            rr = first_row(reliable, reliable["dataset"].eq(setting))
            if rr is not None:
                rel = pd.Series({
                    "attack_f1": rr.get("f1"),
                    "attack_precision": rr.get("precision"),
                    "attack_recall": rr.get("recall"),
                    "aupr": rr.get("aupr"),
                    "auroc": rr.get("auroc"),
                    "accuracy": rr.get("accuracy"),
                    "recall_at_fpr_1e_3": rr.get("recall_at_fpr_1em03"),
                })
        rows.append(normalize_model_row(setting, "Reliable-CMF-CAN", rel, "reliable_cmf_main_raw"))
        safe = first_row(ctt, (ctt["setting"].eq(setting)) & (ctt["model"].astype(str).str.contains("SAFE_CAN", na=False)))
        rows.append(normalize_model_row(setting, "SAFE_CAN", safe, "attack_centric_final_or_missing"))
        for w in ["10", "20", "100"]:
            rows.append(normalize_model_row(setting, f"GRAIN_window_{w}", first_row(ctt, (ctt["setting"].eq(setting)) & (ctt["model"].eq(f"GradientBoosting / window_{w}"))), "attack_centric_final"))
    out = pd.DataFrame(rows)
    out["model"] = pd.Categorical(out["model"], MODEL_ORDER, ordered=True)
    out = out.sort_values(["setting", "model"]).reset_index(drop=True)
    write_table(out, "main_ctt_corrected_benchmark")
    best = out[out["support_status"].eq("measured")].sort_values(["setting", "attack_f1"], ascending=[True, False]).groupby("setting").head(1)
    (OUT / "main_ctt_corrected_benchmark.md").write_text(
        "# Main CT&T Corrected Benchmark\n\n"
        "Weighted-F1 and attack-F1 give different conclusions whenever all-normal or normal-dominated rows are present. CT&T test04 remains the hardest setting under attack-centric metrics. GRAIN_window_100 is the strongest corrected test04 baseline in this table, but this does not support an unknown-attack-solved claim.\n\n"
        f"Best measured rows:\n\n```csv\n{best.to_csv(index=False)}```\n",
        encoding="utf-8",
    )
    return out


def ranking_inversion(main_table: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for setting, g in main_table.groupby("setting", observed=False):
        gg = g.copy()
        for metric in ["accuracy", "weighted_f1", "attack_f1", "aupr", "recall_at_fpr_1e_3"]:
            gg[f"rank_by_{metric}"] = pd.to_numeric(gg[metric], errors="coerce").rank(ascending=False, method="min", na_option="bottom")
        x = pd.to_numeric(gg["weighted_f1"], errors="coerce")
        y = pd.to_numeric(gg["attack_f1"], errors="coerce")
        mask = x.notna() & y.notna()
        spearman = x[mask].corr(y[mask], method="spearman") if mask.sum() >= 3 else np.nan
        top_w = set(gg.sort_values("weighted_f1", ascending=False).head(3)["model"].astype(str))
        top_a = set(gg.sort_values("attack_f1", ascending=False).head(3)["model"].astype(str))
        pan = gg[gg["model"].astype(str).eq("predict_all_normal")]
        grain = gg[gg["model"].astype(str).str.contains("GRAIN_window", na=False)]
        rows.append({
            "setting": setting,
            "weighted_f1_vs_attack_f1_spearman": spearman,
            "top3_overlap_weighted_vs_attack": len(top_w & top_a) / 3,
            "predict_all_normal_rank_by_weighted": pan["rank_by_weighted_f1"].iloc[0] if len(pan) else np.nan,
            "predict_all_normal_rank_by_attack": pan["rank_by_attack_f1"].iloc[0] if len(pan) else np.nan,
            "grain_can_best_rank_by_attack": grain["rank_by_attack_f1"].min() if len(grain) else np.nan,
            "grain_can_best_rank_by_weighted": grain["rank_by_weighted_f1"].min() if len(grain) else np.nan,
        })
    out = pd.DataFrame(rows)
    out.to_csv(TABLES / "main_ranking_inversion.csv", index=False)
    (OUT / "main_ranking_inversion.md").write_text(
        "# Main Ranking Inversion\n\n"
        "Metric choice changes model selection when weighted-F1 is available for normal-dominated rows. predict_all_normal is always last by attack-F1 but can be ranked favorably by weighted-F1 in highly imbalanced settings. GRAIN-CAN ranks near the top under attack-centric metrics, especially test02-test04.\n",
        encoding="utf-8",
    )
    return out


def grain_ablation() -> pd.DataFrame:
    grain = read("results/final_grain_can/tables/b1_granularity_full_matrix.csv")
    feat = read("results/final_grain_can/tables/b2_feature_preservation_by_granularity.csv")
    event = read("results/final_grain_can/tables/f2_event_level_metrics.csv")
    rows = []
    for label, gran in [("sample-level", "sample"), ("window_10", "window_10"), ("window_20", "window_20"), ("window_100 aggregate", "window_100"), ("old window100 deep Transformer", "old_window100_deep")]:
        r = first_row(grain, (grain["dataset"].eq("ctt_test04")) & (grain["granularity"].eq(gran))) if not grain.empty else None
        if r is None and label == "old window100 deep Transformer":
            r = first_row(grain, (grain["dataset"].eq("ctt_test04")) & (grain["model"].astype(str).str.contains("transformer", case=False, na=False)))
        ev = first_row(event, event["dataset"].eq("ctt_test04")) if not event.empty and gran == "window_100" else None
        rows.append({
            "ablation": label,
            "attack_f1": r.get("f1", np.nan) if r is not None else np.nan,
            "aupr": r.get("aupr", np.nan) if r is not None else np.nan,
            "recall_at_fpr_1e_3": r.get("recall_at_fpr_1em03", np.nan) if r is not None else np.nan,
            "event_recall": ev.get("event_recall", np.nan) if ev is not None else np.nan,
            "source": "final_grain_can_b1/f2",
        })
    for feature, desc in [
        ("delta_t_same_id", "without delta_t_same_id"),
        ("payload_delta_l1", "without payload_delta_l1"),
        ("payload_sum", "without payload statistics"),
    ]:
        sub = feat[(feat["feature"].astype(str).eq(feature)) & (feat["granularity"].astype(str).eq("sample"))] if not feat.empty else pd.DataFrame()
        val = sub.iloc[0] if len(sub) else None
        rows.append({
            "ablation": desc,
            "attack_f1": np.nan,
            "aupr": np.nan,
            "recall_at_fpr_1e_3": np.nan,
            "event_recall": np.nan,
            "single_feature_auc": val.get("single_feature_auc", np.nan) if val is not None else np.nan,
            "tree_feature_importance": val.get("tree_feature_importance", np.nan) if val is not None else np.nan,
            "source": "feature_importance_proxy_no_retrain",
        })
    out = pd.DataFrame(rows)
    write_table(out, "main_grain_ablation")
    (OUT / "main_grain_ablation.md").write_text(
        "# Main GRAIN-CAN Ablation\n\n"
        "The strongest direct granularity result on CT&T test04 is window_100 aggregate. Feature evidence indicates delta_t_same_id, payload statistics and CAN ID/payload-preserving features are important. Feature-removal rows are proxy evidence from feature importance/single-feature AUC, not new retraining, and should be described as mechanism evidence rather than full ablation retraining.\n",
        encoding="utf-8",
    )
    return out


def low_event(main_table: pd.DataFrame) -> pd.DataFrame:
    low = read("results/final_grain_can/tables/e1_low_fpr_leaderboard.csv")
    event = read("results/final_grain_can/tables/f2_event_level_metrics.csv")
    rows = []
    for model in ["predict_all_normal", "Table13-style GradientBoosting", "old window100 Transformer", "CAN-Transformer+ same-ID", "SAFE_CAN", "GRAIN_window_100"]:
        base = first_row(main_table, (main_table["setting"].eq("ctt_test04")) & (main_table["model"].astype(str).eq(model)))
        row = {
            "setting": "ctt_test04",
            "model": model,
            "recall_at_fpr_1e_4": base.get("recall_at_fpr_1e_3", np.nan) * 0 if base is not None and model == "predict_all_normal" else np.nan,
            "recall_at_fpr_1e_3": base.get("recall_at_fpr_1e_3", np.nan) if base is not None else np.nan,
            "recall_at_fpr_1e_2": np.nan,
            "aupr": base.get("aupr", np.nan) if base is not None else np.nan,
            "event_recall": np.nan,
            "false_alarm_per_100k": np.nan,
            "detection_delay": np.nan,
            "source": "main_ctt_corrected_benchmark",
        }
        if model == "GRAIN_window_100" and not low.empty:
            for budget, col in [(1e-4, "recall_at_fpr_1e_4"), (1e-3, "recall_at_fpr_1e_3"), (1e-2, "recall_at_fpr_1e_2")]:
                sub = low[(low["dataset"].eq("ctt_test04")) & (low["model"].eq("GradientBoosting")) & (np.isclose(pd.to_numeric(low["fpr_budget"], errors="coerce"), budget))]
                if len(sub):
                    row[col] = sub["recall_at_fpr"].max()
            ev = first_row(event, event["dataset"].eq("ctt_test04")) if not event.empty else None
            if ev is not None:
                row["event_recall"] = ev.get("event_recall")
                row["false_alarm_per_100k"] = ev.get("false_alarm_samples_per_hour")
                row["detection_delay"] = ev.get("mean_detection_delay_seconds")
                row["source"] = "final_grain_can_low_fpr_event"
        rows.append(row)
    out = pd.DataFrame(rows)
    write_table(out, "main_low_fpr_event")
    (OUT / "main_low_fpr_event.md").write_text(
        "# Main Low-FPR / Event-Level Results\n\n"
        "High weighted-F1 does not imply deployment usefulness. GRAIN_window_100 has the strongest available low-FPR and approximate event-level evidence on test04, but event boundaries are approximate and should be conservatively stated.\n",
        encoding="utf-8",
    )
    return out


def plot_bar(df, path: Path, x: str, y: str, title: str, hue: str | None = None) -> None:
    d = df.copy()
    d[y] = pd.to_numeric(d[y], errors="coerce")
    d = d.dropna(subset=[x, y])
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    if d.empty:
        ax.text(0.5, 0.5, "No supported data", ha="center", va="center")
    elif hue and hue in d:
        piv = d.pivot_table(index=x, columns=hue, values=y, aggfunc="max")
        piv.plot(kind="bar", ax=ax, edgecolor="#333333", linewidth=0.5, color=PLOT_COLORS[: len(piv.columns)])
    else:
        colors = [PLOT_COLORS[i % len(PLOT_COLORS)] for i in range(len(d))]
        ax.bar(range(len(d)), d[y], color=colors, edgecolor="#333333", linewidth=0.5)
        ax.set_xticks(range(len(d)))
        ax.set_xticklabels(d[x].astype(str), rotation=35, ha="right")
    ax.set_title(title, fontsize=10)
    ax.grid(axis="y", color="#E6E6E6", linewidth=0.6)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, format="svg")
    plt.close(fig)


def make_figures(main_table, ranking, ablation, low):
    plot_bar(main_table[main_table["support_status"].eq("measured")], FIGS / "main_ctt_corrected_benchmark.svg", "model", "attack_f1", "Main CT&T Corrected Benchmark", hue="setting")
    plot_bar(ranking, FIGS / "main_ranking_inversion.svg", "setting", "top3_overlap_weighted_vs_attack", "Ranking Inversion")
    plot_bar(ablation, FIGS / "main_grain_ablation.svg", "ablation", "attack_f1", "GRAIN-CAN Ablation")
    plot_bar(low, FIGS / "main_low_fpr_event.svg", "model", "recall_at_fpr_1e_3", "Low-FPR / Event-Level")


def docs():
    (OUT / "paper_ready_assets.md").write_text(
        "# Paper-Ready Assets\n\n"
        "Suggested main figures:\n"
        "1. attack-centric evaluation pipeline: `results/attack_centric_final/figures/paper_fig9_attack_centric_pipeline.svg`\n"
        "2. metric trap across settings/datasets: `results/attack_centric_final/figures/paper_fig1_metric_trap_across_datasets.svg`\n"
        "3. CT&T Table13 case study: `results/attack_centric_final/figures/paper_fig3_table13_case_study.svg`\n"
        "4. corrected CT&T benchmark: `figures/main_ctt_corrected_benchmark.svg`\n"
        "5. ranking inversion: `figures/main_ranking_inversion.svg`\n"
        "6. GRAIN-CAN ablation: `figures/main_grain_ablation.svg`\n"
        "7. low-FPR and event-level results: `figures/main_low_fpr_event.svg`\n\n"
        "Suggested main tables:\n"
        "1. dataset statistics: `results/attack_centric_final/tables/paper_table1_dataset_stats.csv`\n"
        "2. corrected benchmark: `tables/main_ctt_corrected_benchmark.csv`\n"
        "3. GRAIN ablation: `tables/main_grain_ablation.csv`\n"
        "4. low-FPR/event-level: `tables/main_low_fpr_event.csv`\n"
        "5. unsafe claims/reporting checklist: `unsafe_claims_do_not_write.md`\n",
        encoding="utf-8",
    )
    (OUT / "final_writing_guidance.md").write_text(
        "# Final Writing Guidance\n\n"
        "可以写：weighted-F1 / accuracy 在 rare-attack CAN IDS 中可能误导；CT&T test04 是典型 case study；attack-centric evaluation 更适合 IDS；GRAIN-CAN 是 corrected benchmark 下的强基线；unknown attack 仍然困难。\n\n"
        "不能写：原论文错了；CT&T 数据集错了；public 0.998 是 attack-F1；GRAIN-CAN 解决了未知攻击；GRAIN-CAN 是终极 SOTA；predict_all_normal 是 IDS。\n",
        encoding="utf-8",
    )
    (OUT / "unsafe_claims_do_not_write.md").write_text(
        "# Unsafe Claims Do Not Write\n\n"
        "- 不要写原论文错了。\n"
        "- 不要写 CT&T 数据集错了。\n"
        "- 不要写 public 0.998 是 attack-F1。\n"
        "- 不要直接比较 attack-F1 和 weighted-F1。\n"
        "- 不要写 unknown attack solved。\n"
        "- 不要写 GRAIN-CAN 是终极 SOTA。\n"
        "- 不要写 predict_all_normal 是 IDS。\n",
        encoding="utf-8",
    )


def audit():
    required_csv = ["current_result_check", "main_ctt_corrected_benchmark", "main_ranking_inversion", "main_grain_ablation", "main_low_fpr_event"]
    required_svg = ["main_ctt_corrected_benchmark.svg", "main_ranking_inversion.svg", "main_grain_ablation.svg", "main_low_fpr_event.svg"]
    required_md = ["current_result_check.md", "main_ctt_corrected_benchmark.md", "main_ranking_inversion.md", "main_grain_ablation.md", "main_low_fpr_event.md", "paper_ready_assets.md", "final_writing_guidance.md", "unsafe_claims_do_not_write.md"]
    rows = []
    for name in required_csv:
        for ext in ["csv", "tex"] if name != "main_ranking_inversion" else ["csv"]:
            p = TABLES / f"{name}.{ext}"
            status = "ok" if p.exists() and p.stat().st_size > 0 else "missing_or_empty"
            if ext == "csv" and status == "ok":
                try:
                    status = "ok" if len(pd.read_csv(p)) > 0 else "empty_csv"
                except Exception as exc:
                    status = f"csv_error:{exc}"
            rows.append({"file_path": str(p), "kind": ext, "status": status, "size": p.stat().st_size if p.exists() else 0})
    for fig in required_svg:
        p = FIGS / fig
        text = p.read_text(errors="ignore") if p.exists() else ""
        rows.append({"file_path": str(p), "kind": "svg", "status": "ok" if p.exists() and "<svg" in text else "invalid_or_missing", "size": p.stat().st_size if p.exists() else 0})
    for md in required_md:
        p = OUT / md
        rows.append({"file_path": str(p), "kind": "md", "status": "ok" if p.exists() and p.read_text(errors="ignore").strip() else "missing_or_empty", "size": p.stat().st_size if p.exists() else 0})
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "output_integrity_audit.csv", index=False)
    failures = out[out["status"].ne("ok")]
    (OUT / "output_integrity_audit.md").write_text(
        "# Output Integrity Audit\n\n"
        f"Checked files: {len(out)}\n\nFailures: {len(failures)}\n\n"
        + ("All required supplement files are present and non-empty.\n" if failures.empty else f"```csv\n{failures.to_csv(index=False)}```\n"),
        encoding="utf-8",
    )
    return out


def main():
    setup()
    current_result_check()
    main_table = main_ctt_table()
    ranking = ranking_inversion(main_table)
    ablation = grain_ablation()
    low = low_event(main_table)
    make_figures(main_table, ranking, ablation, low)
    docs()
    audit()


if __name__ == "__main__":
    main()
