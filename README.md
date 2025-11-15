# AIS Transformer Project

This repository trains a Transformer encoder to predict short term vessel motion from AIS (Automatic Identification System) telemetry.  The project is split into three layers that can also be run independently:

1. **Preprocessing (`preprocessing.py`)** – Downloads raw daily AIS CSV files (optionally zipped), filters them to a geographic bounding box, cleans the tracks, converts speed units, and writes the results to a partitioned Parquet data set keyed by MMSI and trip segment.
2. **Model + Pipeline (`model.py`, `pipeline.py`)** – Defines the Transformer model and implements the data loading, training loop, and evaluation helpers that consume the preprocessed Parquet files.
3. **Entry points (`main.py`, `run_all_preprocessing.py`)** – Provide executable scripts that tie the preprocessing and training steps together.

Set the paths and dates that match your data inside `main.py` or `run_all_preprocessing.py`, run the preprocessing step once, and then train/evaluate the model.
