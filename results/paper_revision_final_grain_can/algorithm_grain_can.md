# Algorithm 1: GRAIN-CAN Training and Inference

Training initializes per-ID history states, processes frames in temporal order, computes same-ID time gaps, same-ID payload differences, payload statistics, and local ID behavior, aggregates fixed windows, trains a supervised classifier, and selects a threshold on validation data if scores are available. Inference freezes feature definitions, classifier, window size, and threshold before test frames are processed. No future frames, test labels, test statistics, test thresholds, or test-window selection are used.
