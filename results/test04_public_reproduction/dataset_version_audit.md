# Dataset Version Audit

The local data root is `data/raw/can-train-and-test/set_01`. No separate `can-train-and-test-v1.5` marker, metadata file, or version manifest was found in the local tree. Therefore exact equivalence to public v1.5 cannot be proven from local files alone.

```csv
dataset_version,source_path,setting,num_files,num_samples,num_positive,num_negative,attack_types,vehicles
local can-train-and-test set_01,data/raw/can-train-and-test/set_01/train_01,train,12,10653140,49557,10603583,DoS;accessory;attack-free;force-neutral;rpm;standstill,unknown_from_directory_name
local can-train-and-test set_01,data/raw/can-train-and-test/set_01/test_01_known_vehicle_known_attack,ctt_test01,8,5702670,63280,5639390,DoS;force-neutral;rpm;standstill,unknown_from_directory_name
local can-train-and-test set_01,data/raw/can-train-and-test/set_01/test_02_unknown_vehicle_known_attack,ctt_test02,8,6447917,164167,6283750,DoS;force-neutral;rpm;standstill,unknown_from_directory_name
local can-train-and-test set_01,data/raw/can-train-and-test/set_01/test_03_known_vehicle_unknown_attack,ctt_test03,12,8635275,20825,8614450,double;fuzzing;interval;speed;systematic;triple,unknown_from_directory_name
local can-train-and-test set_01,data/raw/can-train-and-test/set_01/test_04_unknown_vehicle_unknown_attack,ctt_test04,12,13220555,14244,13206311,double;fuzzing;interval;speed;systematic;triple,unknown_from_directory_name
```
