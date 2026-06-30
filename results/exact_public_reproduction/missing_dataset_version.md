# Missing Dataset Version

No local candidate matches the task-provided can-train-and-test-v1.5 set_01 anchors: total samples 55,582,992 and training samples 11,460,705 with train_02/test05/test06 support.

Current local data found:

```csv
dataset_candidate,path,num_files,total_size_bytes,has_train_01,has_train_02,has_test01,has_test02,has_test03,has_test04,has_test05,has_test06,has_data_extended_xlsx,has_can_ml,file_list_hash,sample_count_train,sample_count_test01,sample_count_test02,sample_count_test03,sample_count_test04,sample_count_total_set01_visible,expected_v15_total_samples,expected_v15_train_samples,columns_train,columns_test04,label_values,attack_types_train,attack_types_test04,vehicle_or_source_fields,status
can-train-and-test,data/raw/can-train-and-test,52,1715703969,True,False,True,True,True,True,False,False,False,False,bcb6ffcaf969d81a,10653140,5702670,6447917,8635275,13220555,44659557,55582992,11460705,timestamp;arbitration_id;data_field;attack,timestamp;arbitration_id;data_field;attack,"{""0"": 10603583, ""1"": 49557}",DoS;accessory;attack-free;force-neutral;rpm;standstill,double;fuzzing;interval;speed;systematic;triple,not_present_in_csv_columns,candidate_original_unverified
set_01,data/raw/can-train-and-test/set_01,52,1715703969,True,False,True,True,True,True,False,False,False,False,bcb6ffcaf969d81a,10653140,5702670,6447917,8635275,13220555,44659557,55582992,11460705,timestamp;arbitration_id;data_field;attack,timestamp;arbitration_id;data_field;attack,"{""0"": 10603583, ""1"": 49557}",DoS;accessory;attack-free;force-neutral;rpm;standstill,double;fuzzing;interval;speed;systematic;triple,not_present_in_csv_columns,candidate_original_unverified
```

To continue exact reproduction, provide the original can-train-and-test release and/or can-train-and-test-v1.5 under one of:
- `CTT_ORIGINAL_ROOT=/path/to/can-train-and-test`
- `CTT_V15_ROOT=/path/to/can-train-and-test-v1.5`
- `data/external/can-train-and-test-v1.5`

Then rerun: `.venv/bin/python -m cmf_can.analysis.exact_public_reproduction`.
