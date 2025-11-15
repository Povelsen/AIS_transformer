"""Convenience script to run preprocessing across a date range."""

from datetime import date
import multiprocessing

from preprocessing import AISDataPreprocessor

# --- 1. CONFIGURE YOUR JOB ---

OUTPUT_PARQUET_ROOT = "path/to/your/parquet_output"
NUM_CORES = max(1, multiprocessing.cpu_count() - 1)

START_DATE = date(2025, 2, 20)
END_DATE = date(2025, 2, 27)


def main() -> None:
    """Process the configured date range using ``AISDataPreprocessor``."""

    preprocessor = AISDataPreprocessor(out_path=OUTPUT_PARQUET_ROOT, num_cores=NUM_CORES)
    preprocessor.process_date_range_parallel(START_DATE, END_DATE)


if __name__ == "__main__":
    main()
