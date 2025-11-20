# --- run_1_preprocessing.py ---
from preprocessing import AISDataPreprocessor
import config
import os

def main():
    print("="*60)
    print("STEP 1: PREPROCESSING")
    print("="*60)

    # Check if raw data exists
    if not os.path.exists(config.DATA_RAW):
        print(f"ERROR: Raw data folder not found at: {config.DATA_RAW}")
        print("Please run 'download_ais_to_folder.py' first.")
        return

    print(f"Input Folder:  {config.DATA_RAW}")
    print(f"Output Folder: {config.PARQUET_OUT}")
    print(f"Target Ship:   {config.SHIP_TYPE}")
    print("-" * 30)

    # Initialize Processor
    preprocessor = AISDataPreprocessor(
        out_path=config.PARQUET_OUT,
        num_cores=config.NUM_CORES,
    )

    # Run
    preprocessor.process_local_csv(config.DATA_RAW)

    print("\nDone! You can now proceed to 'run_2_training.py'.")

if __name__ == "__main__":
    main()