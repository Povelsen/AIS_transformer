import multiprocessing
import os
from datetime import timedelta
from functools import partial

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

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
            print(f"[Processor]: No valid segments found in {file_url}")
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

    def process_date_range_parallel(self, start_date, end_date):
        """
        Public method to download and process all AIS data for a given
        date range using a multiprocessing pool.
        """
        date_strings = self._get_date_range(start_date, end_date)
        
        base_url = "http://aisdata.ais.dk/aisdk-"
        file_urls = [f"{base_url}{d}.zip" for d in date_strings]

        print(f"--- Starting AIS Preprocessing ---")
        print(f"Found {len(file_urls)} days to process.")
        print(f"Using {self.num_cores} CPU cores.")
        print(f"Outputting all data to: {self.out_path}")

        # Create the output directory if it doesn't exist
        os.makedirs(self.out_path, exist_ok=True)

        # Create a processing pool
        process_day_func = partial(self._process_single_file, is_local_csv=False)
        with multiprocessing.Pool(processes=self.num_cores) as pool:
            pool.map(process_day_func, file_urls)

        print("--- All preprocessing jobs complete! ---")

    def process_local_csv(self, local_csv_path):
        """
        Public method to process a single local CSV file.
        """
        print(f"--- Starting Local CSV Preprocessing ---")
        if not os.path.exists(local_csv_path):
            print(f"ERROR: File not found at {local_csv_path}")
            return
            
        # Create the output directory if it doesn't exist
        os.makedirs(self.out_path, exist_ok=True)
        
        # Call the processing function directly
        self._process_single_file(local_csv_path, is_local_csv=True)
        
        print(f"--- Local CSV processing complete! ---")
