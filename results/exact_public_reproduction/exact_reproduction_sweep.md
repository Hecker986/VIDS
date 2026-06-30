# Exact Reproduction Sweep

The sweep is blocked unless a candidate dataset matches the target public fingerprint. Running models on the current local set_01 would be another approximate reproduction, not exact reproduction.

```csv
dataset_version,feature_protocol,model,parameter_set,setting,binary_f1_attack_positive,binary_f1_normal_positive,macro_f1,weighted_f1,accuracy,precision_attack_positive,recall_attack_positive,confusion_matrix,status,blocking_layer,notes
none_aligned,not_run,not_run,not_run,test04,,,,,,,,NA,blocked_dataset_not_aligned,dataset_version/file_manifest,No local candidate matches required public original/v1.5 fingerprint; exact reproduction sweep intentionally not run.
```
