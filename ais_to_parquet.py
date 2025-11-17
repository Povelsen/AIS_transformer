import pandas as pd
import pyarrow
import pyarrow.parquet
#from sklearn.cluster import KMeans

Test = False
show = False
save = True

if Test:

    file_path = "aisdk-2025-03-01.csv"

    # Load your ports file
    df = pd.read_csv(file_path, nrows=1)

    print("Columns:", df.columns.tolist())
    print("Ship type sample:", df["Ship type"].head())


    exit()

def fn(file_path, out_path):
    dtypes = {
        "MMSI": "object",
        "SOG": float,
        "COG": float,
        "Longitude": float,
        "Latitude": float,
        "# Timestamp": "object",
        "Type of mobile": "object",
        "Ship type": "object",
        "Cargo type": "object",
    }
    usecols = list(dtypes.keys())
    df = pd.read_csv(file_path, usecols=usecols, dtype=dtypes)

    # Remove errors
    bbox = [60, 0, 50, 20]
    north, west, south, east = bbox
    df = df[(df["Latitude"] <= north) & (df["Latitude"] >= south) & (df["Longitude"] >= west) & (
            df["Longitude"] <= east)]

    # --- Keep only Class A vessels ---
    df["Type of mobile"] = df["Type of mobile"].astype(str).str.strip()
    df = df[df["Type of mobile"] == "Class A"]
    # Keep only Cargo or Tanker ship types ---
    df["Ship type"] = df["Ship type"].astype(str).str.strip()
    df = df[df["Ship type"].isin(["Cargo", "Tanker"])]
    # Drop the Type of mobile column (you only need Ship type now) ---
    df = df.drop(columns=["Type of mobile"])

    df = df[df["MMSI"].str.len() == 9]  # Adhere to MMSI format
    df = df[df["MMSI"].str[:3].astype(int).between(200, 775)]  # Adhere to MID standard

    df = df.rename(columns={"# Timestamp": "Timestamp"})
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], format="%d/%m/%Y %H:%M:%S", errors="coerce")

    df = df.drop_duplicates(["Timestamp", "MMSI", ], keep="first")

    def track_filter(g):
        len_filt = len(g) > 256  # Min required length of track/segment
        sog_filt = 1 <= g["SOG"].max() <= 50  # Remove stationary tracks/segments
        time_filt = (g["Timestamp"].max() - g["Timestamp"].min()).total_seconds() >= 60 * 60  # Min required timespan
        return len_filt and sog_filt and time_filt

    # Track filtering
    df = df.groupby("MMSI").filter(track_filter)
    df = df.sort_values(['MMSI', 'Timestamp'])

    # Divide track into segments based on timegap
    df['Segment'] = df.groupby('MMSI')['Timestamp'].transform(
        lambda x: (x.diff().dt.total_seconds().fillna(0) >= 15 * 60).cumsum())  # Max allowed timegap

    # Segment filtering
    df = df.groupby(["MMSI", "Segment"]).filter(track_filter)
    df = df.reset_index(drop=True)

    #
    knots_to_ms = 0.514444
    df["SOG"] = knots_to_ms * df["SOG"]

    # Clustering
    # kmeans = KMeans(n_clusters=48, random_state=0)
    # kmeans.fit(df[["Latitude", "Longitude"]])
    # df["Geocell"] = kmeans.labels_
    # centers = kmeans.cluster_centers_
    # "Latitude": center[0],
    # "Longitude": center[1],

    # df["Date"] = df["Timestamp"].dt.strftime("%Y-%m-%d")

    if show:
        print("\n--- CLEANED DATA SAMPLE ---")
        print(df)
        print("\nTotal rows after cleaning:", len(df))

    if save:
        # Save as parquet file with partitions
        table = pyarrow.Table.from_pandas(df, preserve_index=False)
        pyarrow.parquet.write_to_dataset(
            table,
            root_path=out_path,
            partition_cols=["MMSI", "Segment"]
        )
        print(f"Saved Parquet dataset to {out_path}")


if __name__ == "__main__":
    dates = ["2025-02-12", "2025-02-13", "2025-02-14", "2025-02-15", "2025-02-16", "2025-02-17", "2025-02-18"]
    for date in dates:
        input_file = f"data_raw_csv/aisdk-{date}.csv"
        output_dir = f"data_cleaned_parquet/{date}_processed"
        fn(input_file, output_dir)


