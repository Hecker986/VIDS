# Recommended Main Model

Do not replace all results with untrained Reliable-CMF-CAN. Use the original CMF-CAN as the supervised encoder, and define the forward-looking main model as **Reliable-CMF-CAN + normality/adaptive policy** once trained.

Current main metrics should be AUPR and Recall@FPR for deployment, with F1 reported but not used as the only headline. The strongest paper line is reliable low-FPR/open-world CAN IDS, not universal F1 superiority.
