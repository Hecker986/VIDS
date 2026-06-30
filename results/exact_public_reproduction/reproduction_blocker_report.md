# Reproduction Blocker Report

No exact public reproduction can be claimed from the current workspace.

1. Original data version: not fingerprint-aligned to a public manifest in this workspace.
2. v1.5 data version: missing or not aligned; no candidate has v1.5 sample-count anchors and train_02/test05/test06 support.
3. Official feature table: not present locally.
4. File manifest: current local set_01 does not include v1.5-required subsets and sample counts do not match 55,582,992 total / 11,460,705 train.
5. Field format: local CSV columns are timestamp, arbitration_id, data_field, attack; no extra metadata proving public protocol equivalence.
6. Metric definition: exact public confusion matrices/positive-label convention are unavailable locally.
7. Required next input: provide exact original release and/or v1.5 under `CTT_ORIGINAL_ROOT` or `CTT_V15_ROOT`.

Until this is resolved, do not claim the method fails to match public 0.998 and do not claim the public result is wrong.
