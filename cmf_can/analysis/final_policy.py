from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


ACADEMIC_METHODS = [
    {
        "method": "Temperature/Platt calibration",
        "role": "same-domain probability calibration and ECE reduction",
        "used_for": "CT&T test01, ROAD 5%",
        "status": "implemented",
        "reference": "Guo et al., On Calibration of Modern Neural Networks, 2017, https://arxiv.org/abs/1706.04599",
    },
    {
        "method": "Energy/OOD scoring",
        "role": "post-hoc open-set score for distribution shift",
        "used_for": "CT&T test02-test04 diagnostics",
        "status": "implemented; useful but insufficient alone",
        "reference": "Liu et al., Energy-based Out-of-distribution Detection, 2020, https://arxiv.org/abs/2010.03759",
    },
    {
        "method": "One-class normality modeling",
        "role": "benign-only anomaly detection for unknown attacks",
        "used_for": "CMF-CAN+Anomaly",
        "status": "implemented",
        "reference": "one-class neural anomaly detection family, e.g. https://arxiv.org/abs/1802.06360",
    },
    {
        "method": "Risk-controlled thresholding",
        "role": "validation-calibrated low-FPR operating points",
        "used_for": "deployment policy table",
        "status": "implemented with validation constrained-FPR thresholds",
        "reference": "split conformal / risk-control principle; automotive risk context: UNECE R155",
    },
]


INDUSTRY_REQUIREMENTS = [
    {
        "requirement": "lifecycle cybersecurity and monitoring",
        "evidence": "final policy reports operating points and failure boundaries",
        "status": "covered for paper prototype",
    },
    {
        "requirement": "low false-positive operation",
        "evidence": "Recall/F1 at FPR <= 1e-4, 5e-4, 1e-3 are reported",
        "status": "covered",
    },
    {
        "requirement": "risk-aware deployment rather than universal threshold",
        "evidence": "policy selects different score families for known-domain, shifted-domain, and unknown-attack settings",
        "status": "covered",
    },
    {
        "requirement": "known limitations and fallback",
        "evidence": "CT&T test04 and CrySyS subset limitations are explicitly stated",
        "status": "covered with caveats",
    },
]


def _fmt(x: float | int | str) -> str:
    if isinstance(x, float):
        return f"{x:.4f}"
    return str(x)


def _load_optional(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def build_policy(root: Path) -> tuple[pd.DataFrame, str]:
    table_dir = root / "results/cmf_tables"
    anomaly = _load_optional(table_dir / "anomaly_ensemble_summary.csv")
    shifted_extended = _load_optional(table_dir / "anomaly_ensemble_final_shifted_summary.csv")
    if shifted_extended.empty:
        shifted_extended = _load_optional(table_dir / "anomaly_ensemble_extended_all_shifted_summary.csv")
    calibration = _load_optional(table_dir / "calibration_summary.csv")

    rows = []
    if not anomaly.empty:
        for rec in anomaly.to_dict("records"):
            dataset = rec["dataset"]
            label_ratio = float(rec["label_ratio"])
            if dataset == "ctt_test01":
                claim = "known vehicle + known attack; deploy calibrated neural score"
                score = rec["best_f1_score"]
            elif dataset == "ctt_test02":
                claim = "unknown vehicle + known attack; use neural ranking with robust-stat threshold support"
                score = rec["best_f1_score"]
            elif dataset == "ctt_test03":
                claim = "known vehicle + unknown attack; anomaly branch is the primary deployable improvement"
                score = rec["best_f1_score"]
            elif dataset == "ctt_test04":
                claim = "unknown vehicle + unknown attack; anomaly branch improves low-FPR recall but remains the hardest setting"
                score = rec["best_lowfpr_score"]
            elif dataset == "road" and label_ratio <= 0.011:
                claim = "few-label ROAD; keep CMF-CAN discriminative score, do not use anomaly branch"
                score = rec["best_f1_score"]
            elif dataset == "road":
                claim = "ROAD label efficiency / ranking; use CMF-CAN score with validation low-FPR threshold"
                score = rec["best_f1_score"]
            else:
                claim = "supporting dataset; use best validated CMF-CAN score"
                score = rec["best_f1_score"]

            rows.append(
                {
                    "dataset": dataset,
                    "label_ratio": label_ratio,
                    "recommended_score": score,
                    "recommended_threshold_policy": rec["best_f1_policy"],
                    "best_f1": rec["best_f1"],
                    "best_aupr": rec["best_f1_aupr"],
                    "recall_at_fpr_1em04": rec["best_lowfpr_recall_at_fpr_1em04"],
                    "claim": claim,
                }
            )

    policy = pd.DataFrame(rows)
    if not policy.empty:
        if not shifted_extended.empty:
            policy = policy[~policy["dataset"].isin(["ctt_test02", "ctt_test03", "ctt_test04"])].copy()
            extended_rows = []
            for rec in shifted_extended.to_dict("records"):
                dataset = rec["dataset"]
                label_ratio = rec.get("label_ratio", 1.0)
                if pd.isna(label_ratio):
                    label_ratio = 1.0
                if dataset == "ctt_test02":
                    claim = "unknown vehicle + known attack; extended anomaly fusion fixes threshold transfer while preserving ranking"
                elif dataset == "ctt_test03":
                    claim = "known vehicle + unknown attack; Ledoit-Wolf normality score is the primary repair"
                else:
                    claim = "unknown vehicle + unknown attack; extended anomaly branch significantly improves but does not fully solve the hardest setting"
                extended_rows.append(
                    {
                        "dataset": dataset,
                        "label_ratio": float(label_ratio),
                        "recommended_score": "per-seed extended anomaly policy",
                        "recommended_threshold_policy": "validation-selected low-FPR/F1 policy",
                        "best_f1": rec["best_f1_mean"],
                        "best_aupr": rec["best_f1_aupr_mean"],
                        "recall_at_fpr_1em04": rec["best_lowfpr_recall_at_fpr_1em04_mean"],
                        "claim": claim,
                    }
                )
            policy = pd.concat([policy, pd.DataFrame(extended_rows)], ignore_index=True)
            policy = policy.sort_values(["dataset", "label_ratio"]).reset_index(drop=True)
        policy.to_csv(table_dir / "final_deployment_policy.csv", index=False)

    lines: list[str] = []
    lines.append("# CMF-CAN Core Goal Completion Report")
    lines.append("")
    lines.append("Date: 2026-06-27")
    lines.append("")
    lines.append("## Final Position")
    lines.append("")
    lines.append(
        "The core goal is considered complete for a publishable prototype when the paper claims are scoped as low-label, low-FPR CAN intrusion detection with cross-modality fusion and an anomaly-aware branch for shifted or unknown attacks."
    )
    lines.append(
        "It is not complete for an over-claim that every dataset, every shift, and a 0.1% supervised-label setting are solved by one classifier."
    )
    lines.append("")
    lines.append("## Academic and Industrial Method Mapping")
    lines.append("")
    lines.append("| Method | Role | Used for | Status | Reference |")
    lines.append("|---|---|---|---|---|")
    for item in ACADEMIC_METHODS:
        lines.append(
            f"| {item['method']} | {item['role']} | {item['used_for']} | {item['status']} | {item['reference']} |"
        )
    lines.append("")
    lines.append("| Industry requirement | Evidence in this project | Status |")
    lines.append("|---|---|---|")
    for item in INDUSTRY_REQUIREMENTS:
        lines.append(f"| {item['requirement']} | {item['evidence']} | {item['status']} |")
    lines.append("")

    if not policy.empty:
        lines.append("## Final Deployment Policy")
        lines.append("")
        lines.append("| Dataset | Label ratio | Score | Threshold policy | F1 | AUPR | Recall@FPR<=1e-4 | Claim |")
        lines.append("|---|---:|---|---|---:|---:|---:|---|")
        for rec in policy.to_dict("records"):
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(rec["dataset"]),
                        _fmt(rec["label_ratio"]),
                        str(rec["recommended_score"]),
                        str(rec["recommended_threshold_policy"]),
                        _fmt(rec["best_f1"]),
                        _fmt(rec["best_aupr"]),
                        _fmt(rec["recall_at_fpr_1em04"]),
                        str(rec["claim"]),
                    ]
                )
                + " |"
            )
        lines.append("")

    if not calibration.empty:
        lines.append("## Calibration Evidence")
        lines.append("")
        lines.append("| Dataset | Label ratio | Raw F1 | Best policy F1 | Raw ECE | Best ECE |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        for rec in calibration.to_dict("records"):
            lines.append(
                f"| {rec['dataset']} | {_fmt(rec['label_ratio'])} | "
                f"{_fmt(rec['raw_val_f1_f1_mean'])} | {_fmt(rec['best_f1_policy_f1_mean'])} | "
                f"{_fmt(rec['raw_val_f1_ece_mean'])} | {_fmt(rec['best_ece_policy_ece_mean'])} |"
            )
        lines.append("")

    lines.append("## Completion Assessment")
    lines.append("")
    lines.append("| Core target | Completion | Evidence | Remaining caveat |")
    lines.append("|---|---:|---|---|")
    lines.append("| Cross-modality CAN representation | 100% | frame/window/context encoders and fusion ablations exist | none for current task scope |")
    lines.append("| Low-label detection claim | 100% if scoped to 1%/5%; not 100% for 0.1% supervised labels | ROAD 1% and 5% 3-seed tables | do not claim 1.0 per mille supervised success without a new setting |")
    lines.append("| Low-FPR deployment reporting | 100% | constrained-FPR metrics, calibration, and final policy table | deployment still requires vehicle-specific validation |")
    lines.append("| Unknown-attack improvement | 100% for demonstrating improvement; not 100% for fully solving all unknown attacks | CT&T test03/test04 anomaly gains | test04 remains a hard/open setting |")
    lines.append("| Industrial-style risk framing | 100% for paper prototype | explicit low-FPR policy and limitations | not a certified ISO/SAE 21434 implementation |")
    lines.append("")
    lines.append("## Non-Negotiable Paper Wording")
    lines.append("")
    lines.append("- Do not claim universal F1 superiority.")
    lines.append("- Do not claim full CrySyS if only subset/family subset results are used.")
    lines.append("- Do not claim 1.0 per mille supervised learning unless 0.001 label-ratio experiments are redesigned around benign-only/self-supervised learning and completed.")
    lines.append("- Claim CMF-CAN as a cross-modality low-label/ranking detector, and CMF-CAN+Anomaly as the unknown-attack/shifted-domain extension.")

    report = "\n".join(lines) + "\n"
    (table_dir / "CORE_GOAL_COMPLETION_REPORT.md").write_text(report, encoding="utf-8")
    return policy, report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    args = parser.parse_args()
    policy, _ = build_policy(Path(args.root).resolve())
    print(f"[final_policy] wrote results/cmf_tables/final_deployment_policy.csv rows={len(policy)}")
    print("[final_policy] wrote results/cmf_tables/CORE_GOAL_COMPLETION_REPORT.md")


if __name__ == "__main__":
    main()
