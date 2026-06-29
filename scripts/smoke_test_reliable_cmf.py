from __future__ import annotations

from pathlib import Path

import torch

from cmf_can.models.cmf import build_model


def main() -> None:
    out = Path("results/reliable_cmf/logs/smoke_test_reliable_cmf.log")
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(42)
    model = build_model("reliable_cmf_can")
    model.train()
    batch = {
        "can_id": torch.randint(0, 4097, (4, 100), dtype=torch.long),
        "payload": torch.randint(0, 255, (4, 100, 8), dtype=torch.long),
        "frame_numeric": torch.randn(4, 100, 10),
        "window_stats": torch.randn(4, 26),
        "id_context": torch.randn(4, 100, 18),
        "label": torch.tensor([0, 1, 0, 1], dtype=torch.long),
    }
    logits, aux = model(batch, return_aux=True)
    required = [
        "gate_frame",
        "gate_window",
        "gate_context",
        "reliability_frame",
        "reliability_window",
        "reliability_context",
        "context_shift_score",
        "topk_score",
    ]
    lines = [f"logits_shape={tuple(logits.shape)}"]
    if logits.shape != (4, 2):
        raise AssertionError(f"unexpected logits shape: {tuple(logits.shape)}")
    for key in required:
        if key not in aux:
            raise AssertionError(f"missing aux key: {key}")
        if torch.isnan(aux[key]).any():
            raise AssertionError(f"NaN in aux key: {key}")
        lines.append(f"{key}_mean={float(aux[key].detach().mean()):.6f}")
    if torch.isnan(logits).any():
        raise AssertionError("NaN in logits")
    loss = torch.nn.functional.cross_entropy(logits, batch["label"])
    loss.backward()
    grad_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            grad_norm += float(p.grad.detach().norm())
    if grad_norm <= 0:
        raise AssertionError("no gradient flowed")
    lines.append(f"loss={float(loss.detach()):.6f}")
    lines.append(f"grad_norm_sum={grad_norm:.6f}")
    lines.append("status=ok")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
