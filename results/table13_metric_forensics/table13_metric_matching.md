# Table 13 Metric Matching

Most consistent available hypothesis from task-provided approximate F1 anchors: `weighted_f1`.

Because the exact Table 13 cells are not present in the workspace, matching is performed against task-stated high-F1 anchors for BIRCH, GradientBoosting and MLP. The evidence is still enough to show that local attack-positive F1 is far from 0.998, while normal-positive/weighted/accuracy-like metrics can be high under extreme imbalance.

```csv
model,local_model,feature_protocol,negative_protocol,seed,reported_f1,reported_precision,reported_recall,status,abs_diff_reported_f1_vs_attack_f1,abs_diff_reported_f1_vs_normal_f1,abs_diff_reported_f1_vs_macro_f1,abs_diff_reported_f1_vs_weighted_f1,abs_diff_reported_f1_vs_accuracy,abs_diff_reported_precision_vs_attack_precision,abs_diff_reported_precision_vs_normal_precision,abs_diff_reported_recall_vs_attack_recall,abs_diff_reported_recall_vs_normal_recall,abs_diff_reported_recall_vs_accuracy,best_f1_hypothesis,best_f1_hypothesis_abs_diff
GradientBoosting,GradientBoosting,SAFE_CAN,A_capped,42.0,0.998,,,matched_when_reported_target_available,0.8023564356435529,0.0014619548896651624,0.4004472403769438,0.000595910373776376,0.0009246291097463333,,,,,,weighted_f1,0.000595910373776376
GradientBoosting,GradientBoosting,P7_public_plus_delta,A_capped,42.0,0.998,,,matched_when_reported_target_available,0.8020797772600643,0.0014644542127651983,0.4003076615236495,0.0005987050794572513,0.0009296213358667149,,,,,,weighted_f1,0.0005987050794572513
GradientBoosting,GradientBoosting,SAFE_CAN,C_5x_negative_cap,42.0,0.998,,,matched_when_reported_target_available,0.5183757293989036,0.0015848225916161862,0.25839545340364367,0.0010246102478415064,0.0011703071467121795,,,,,,weighted_f1,0.0010246102478415064
GradientBoosting,GradientBoosting,SAFE_CAN,B_2x_negative_cap,42.0,0.998,,,matched_when_reported_target_available,0.5167955686014447,0.0015852382673300358,0.25760516516705734,0.0010267279617461877,0.0011711391843988173,,,,,,weighted_f1,0.0010267279617461877
MLP,MLP,P7_public_plus_delta,A_capped,42.0,0.998,,,matched_when_reported_target_available,0.986820625035067,0.0033611369701861404,0.49509088100262655,0.004420729169454352,0.008664453950685158,,,,,,normal_f1,0.0033611369701861404
MLP,MLP,SAFE_CAN,B_2x_negative_cap,42.0,0.998,,,matched_when_reported_target_available,0.989869918699187,0.008340604523886408,0.4991052616115367,0.009398117128396,0.018467824535354227,,,,,,normal_f1,0.008340604523886408
GradientBoosting,GradientBoosting,P1_public_default,B_2x_negative_cap,42.0,0.998,,,matched_when_reported_target_available,0.9911159890894922,0.009703143325622832,0.5004095662075575,0.010760530445532712,0.02113367328376159,,,,,,normal_f1,0.009703143325622832
MLP,MLP,SAFE_CAN,A_capped,42.0,0.998,,,matched_when_reported_target_available,0.9915435039392654,0.011281459053420817,0.5014124814963431,0.012337606285314884,0.024212515283964953,,,,,,normal_f1,0.011281459053420817
MLP,MLP,SAFE_CAN,C_5x_negative_cap,42.0,0.998,,,matched_when_reported_target_available,0.9943819043715436,0.01587499029359596,0.5051284473325698,0.016929246524451935,0.03311993255956347,,,,,,normal_f1,0.01587499029359596
GradientBoosting,GradientBoosting,P1_public_default,C_5x_negative_cap,42.0,0.998,,,matched_when_reported_target_available,0.9936566827304337,0.016869627412201882,0.5052631550713178,0.017922030644589193,0.03503732558882744,,,,,,normal_f1,0.016869627412201882
GradientBoosting,GradientBoosting,P1_public_default,A_capped,42.0,0.998,,,matched_when_reported_target_available,0.9940740673173402,0.01908673554510576,0.506580401431223,0.020137199731651845,0.03929917390003668,,,,,,normal_f1,0.01908673554510576
MLP,MLP,P5_arbitration_payload_only,A_capped,42.0,0.998,,,matched_when_reported_target_available,0.993052991091863,0.03341932294486216,0.5132361570183626,0.034453244892094004,0.06640378486379717,,,,,,normal_f1,0.03341932294486216
GradientBoosting,GradientBoosting,P5_arbitration_payload_only,A_capped,42.0,0.998,,,matched_when_reported_target_available,0.9917067809287672,0.03765792674981838,0.5146823538392928,0.038685831544959415,0.07427191142883172,,,,,,normal_f1,0.03765792674981838
GradientBoosting,GradientBoosting,P3_no_subdivision,A_capped,42.0,0.998,,,matched_when_reported_target_available,0.9928820465442265,0.047598082708602996,0.5202400646264147,0.04861654409550209,0.09248574587072933,,,,,,normal_f1,0.047598082708602996
MLP,MLP,P1_public_default,A_capped,42.0,0.998,,,matched_when_reported_target_available,0.9958455130243867,0.05229996734793618,0.5240727401861615,0.05331655574022498,0.1009952222126832,,,,,,normal_f1,0.05229996734793618
MLP,MLP,P1_public_default,C_5x_negative_cap,42.0,0.998,,,matched_when_reported_target_available,0.9963168415520397,0.0694378403544067,0.5328773409532233,0.07043647199226599,0.13133449314344203,,,,,,normal_f1,0.0694378403544067
MLP,MLP,P1_public_default,B_2x_negative_cap,42.0,0.998,,,matched_when_reported_target_available,0.9962680320793491,0.07058234221860116,0.5334251871489751,0.07157968816711657,0.1333254080483005,,,,,,normal_f1,0.07058234221860116
MLP,MLP,P3_no_subdivision,A_capped,42.0,0.998,,,matched_when_reported_target_available,0.9960595290288682,0.25014521395858613,0.6231023714937272,0.25094887186128856,0.4005835526572068,,,,,,normal_f1,0.25014521395858613
BIRCH,BIRCH,SAFE_CAN,A_capped,42.0,0.998,,,matched_when_reported_target_available,0.995514829103819,0.6818476896675701,0.8386812593856946,0.6821856387728936,0.8094231210414389,,,,,,normal_f1,0.6818476896675701
BIRCH,BIRCH,P1_public_default,A_capped,42.0,0.998,,,matched_when_reported_target_available,0.9957280627442776,0.7187926641984259,0.8572603634713518,0.7190910380425564,0.8349466334809696,,,,,,normal_f1,0.7187926641984259
```
