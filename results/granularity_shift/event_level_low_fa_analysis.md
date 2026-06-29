# Event-Level Low False Alarm Analysis

Results:

```csv
dataset,model,event_recall,events,mean_detection_delay_windows,median_detection_delay_windows,false_alarm_windows,false_alarm_windows_per_100k,note
ctt_test02,window100_transformer,1.0,593,0.0,0.0,58867,91302.0550601008,contiguous positive windows; no official event boundary files
ctt_test04,window100_transformer,1.0,463,0.0021598272138228,0.0,128693,97347.20121028744,contiguous positive windows; no official event boundary files
road,window100_transformer,0.1641791044776119,67,0.0,0.0,2,3.544905085166345,contiguous positive windows; no official event boundary files
road,can_transformer_plus_sameid,0.1641791044776119,67,0.0,0.0,2,3.544905085166345,contiguous positive windows; no official event boundary files
```

No official event boundaries were found. Events are approximated as contiguous positive windows; false alarms are reported per 100k windows rather than per hour.
