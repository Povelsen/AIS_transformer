import multiprocessing
import os
from datetime import timedelta
from functools import partial

import config
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path

class AISDataPreprocessor:
    """
    Handles downloading, processing, and saving raw AIS data into
    a partitioned Parquet dataset.
    """
    def __init__(self, out_path, num_cores=None):
        self.out_path = out_path
        default_cores = max(1, multiprocessing.cpu_count() - 1)
        self.num_cores = num_cores if num_cores else default_cores
        
        # Bounding Box for Denmark
        self.bbox = [60, 0, 50, 20] # North, West, South, East
        
        # Data types for efficient loading
        self.dtypes = {
            "MMSI": "object",
            "SOG": float,
            "COG": float,
            "Longitude": float,
            "Latitude": float,
            "# Timestamp": "object",
            "Type of mobile": "object",
            "Ship type": "object",        # <- we need this column for filtering
        }
        
    def _get_date_range(self, start_date, end_date):
        """Generates a list of date strings between two dates."""
        dates = []
        delta = end_date - start_date
        for i in range(delta.days + 1):
            day = start_date + timedelta(days=i)
            dates.append(day.strftime('%Y-%m-%d'))
        return dates

    def _process_single_file(self, file_url, is_local_csv=False):
        """
        The core logic to process one day's worth of AIS data.
        Can be run in parallel by the pool or standalone.
        """
        if is_local_csv:
            print(f"[Processor]: Starting local file {file_url}")
        else:
            print(f"[Processor]: Starting URL {file_url}")
            
        usecols = list(self.dtypes.keys())
        
        try:
            if is_local_csv:
                df = pd.read_csv(file_url, usecols=usecols, dtype=self.dtypes)
            else:
                df = pd.read_csv(file_url, usecols=usecols, dtype=self.dtypes, compression='zip')

        except Exception as e:
            print(f"ERROR: Could not process {file_url}. Reason: {e}")
            return

        # 1. Bounding Box Filter
        north, west, south, east = self.bbox
        df = df[(df["Latitude"] <= north) & (df["Latitude"] >= south) & 
                (df["Longitude"] >= west) & (df["Longitude"] <= east)]

        # 1b. Filter by ship type: keep only Cargo ships       # <<< NEW
        if "Ship type" in df.columns:
            df["Ship type"] = df["Ship type"].fillna("").str.strip()
            df = df[df["Ship type"].str.lower() == "cargo"]
        else:
            # If for some reason the column isn't present, drop everything
            # so we don't accidentally mix in non-cargo traffic.
            print(f"[Processor]: 'Ship type' column missing in {file_url}, skipping.")
            return

        # 2. Clean and Validate
        df = df[df["Type of mobile"].isin(["Class A", "Class B"])].drop(columns=["Type of mobile"])
        df = df[df["MMSI"].str.len() == 9]  # Adhere to MMSI format
        df = df[df["MMSI"].str[:3].astype(int).between(200, 775)]  # Adhere to MID standard
        
        df = df.rename(columns={"# Timestamp": "Timestamp"})
        df["Timestamp"] = pd.to_datetime(df["Timestamp"], format="%d/%m/%Y %H:%M:%S", errors="coerce")

        df = df.drop_duplicates(["Timestamp", "MMSI"], keep="first")
        df = df.dropna(subset=["Timestamp", "SOG", "COG", "Latitude", "Longitude"])

        # 3. Track Filtering
        def track_filter(g):
            len_filt = len(g) > 256  # Min required length
            sog_filt = 1 <= g["SOG"].max() <= 50  # Remove stationary
            time_filt = (g["Timestamp"].max() - g["Timestamp"].min()).total_seconds() >= 60 * 60  # Min 1 hour
            return len_filt and sog_filt and time_filt

        df = df.groupby("MMSI").filter(track_filter)
        df = df.sort_values(['MMSI', 'Timestamp'])

        # 4. Segment Filtering (Identify individual "trips")
        # A new segment is created if the time gap is > 15 minutes
        df['Segment'] = df.groupby('MMSI')['Timestamp'].transform(
            lambda x: (x.diff().dt.total_seconds().fillna(0) >= 15 * 60).cumsum())

        # 5. Re-apply the filter to the *segments*
        df = df.groupby(["MMSI", "Segment"]).filter(track_filter)
        df = df.reset_index(drop=True)

        if df.empty:
            print(f"[Processor]: No valid cargo segments found in {file_url}")
            return

        # 6. Final Conversion
        knots_to_ms = 0.514444
        df["SOG"] = knots_to_ms * df["SOG"]

        # 7. Save to Parquet
        table = pa.Table.from_pandas(df, preserve_index=False)
        pq.write_to_dataset(
            table,
            root_path=self.out_path,
            partition_cols=["MMSI", "Segment"],
            existing_data_behavior='overwrite_or_ignore' # Appends new data
        )
        if is_local_csv:
            print(f"[Processor]: Finished local file {file_url}")
        else:
            print(f"[Processor]: Finished URL {file_url}")

    def process_local_csv(self, local_csv_path):
        """Public method to process one or more local CSV files.

        ``local_csv_path`` may point to a single CSV file or a directory that
        contains multiple CSV files downloaded from the AIS Danish archive. All
        discovered files are processed sequentially and appended to the Parquet
        dataset under ``self.out_path``.
        """

        print("--- Starting Local CSV Preprocessing ---")

        source_path = Path(local_csv_path)
        if not source_path.exists():
            print(f"ERROR: File or directory not found at {source_path}")
            return

        if source_path.is_dir():
            csv_files = sorted(p for p in source_path.glob("*.csv") if p.is_file())
            if not csv_files:
                print(f"ERROR: No CSV files found in directory {source_path}")
                return
            targets = csv_files
        else:
            if source_path.suffix.lower() != ".csv":
                print(f"ERROR: Expected a .csv file, got {source_path.suffix} at {source_path}")
                return
            targets = [source_path]

        os.makedirs(self.out_path, exist_ok=True)

        for csv_path in targets:
            self._process_single_file(str(csv_path), is_local_csv=True)

        print("--- Local CSV processing complete! ---")
