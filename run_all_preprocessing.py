"""Convenience script to run preprocessing for remote or local AIS data."""

from datetime import date
import multiprocessing

from preprocessing import AISDataPreprocessor


# --- 1. CONFIGURE YOUR JOB ---

# Where the Parquet dataset should be written
OUTPUT_PARQUET_ROOT = "data/ais_parquet_data"

# Parallelism for remote downloads
NUM_CORES = max(1, multiprocessing.cpu_count() - 1)

# Remote download window (ignored when using local CSV files)
START_DATE = date(2025, 2, 20)
END_DATE = date(2025, 2, 27)

# Local CSV ingestion (set to ``None`` to skip)
LOCAL_CSV_SOURCE = "data"  # Either a directory of CSVs or a single CSV file


def main() -> None:
    """Run preprocessing on downloaded ZIPs or locally stored CSV files."""

    preprocessor = AISDataPreprocessor(out_path=OUTPUT_PARQUET_ROOT, num_cores=NUM_CORES)

    if LOCAL_CSV_SOURCE:
        preprocessor.process_local_csv(LOCAL_CSV_SOURCE)
    else:
        preprocessor.process_date_range_parallel(START_DATE, END_DATE)


if __name__ == "__main__":
    main()
