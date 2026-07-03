# CIVS 2026 Scope and Format Compliance Check

Source checked: user-provided CIVS 2026 call text and the public conference URL `https://ccf.org.cn/civs2026`. The public page is JavaScript-rendered in this environment, so the detailed checklist uses the user's supplied call-for-papers excerpt.

## Scope Fit

The paper fits the listed **Automotive Safety and Security** scope:

- **In-Vehicle Network Security**: the core topic is CAN intrusion detection and attack-centric evaluation.
- **Vehicle-Mounted AI Safety and Explainability**: the work audits metric-driven model selection, claim boundaries, and explainable feature-preserving evidence.
- Adjacent: **Automotive Safety Verification and Validation**, because the paper validates whether IDS evidence is aligned with attack detection rather than normal-class recognition.

## Paper Type

- Suitable type: **Research Paper**.
- Language: English, which is allowed by the call.
- Recommended framing: `attack-centric evaluation framework + corrected CAN IDS benchmark + feature-preserving GRAIN-CAN baseline`.

## Page Limit Risk

- The call says Research Papers are recommended to be no more than 16 pages.
- Local fallback compilation currently produces 19 pages because this environment does not have the official CIVS/LNCS class installed and uses the built-in article fallback.
- This remains a submission risk. The source now has tighter float spacing, shorter captions, and top-aligned float pages, but final page count must be rechecked with the official CIVS template package.

## Required Claim Boundary

- Do not claim unknown attacks are solved.
- Do not claim universal external-dataset dominance.
- Do not accuse CT&T or original authors; state metric ambiguity and corrected evaluation evidence.
