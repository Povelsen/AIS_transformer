import multiprocessing
from datetime import date, timedelta
from functools import partial
from preprocessing import fn  # Import your function

# --- 1. CONFIGURE YOUR JOB ---

# This is the single output directory all days will be written to
OUTPUT_PARQUET_ROOT = "path/to/your/parquet_output" 

# Set how many CPU cores to use (e.g., all cores minus one)
NUM_CORES = multiprocessing.cpu_count() - 1 

# --- 2. GENERATE THE LIST OF FILES ---
def get_date_range(start_date, end_date):
    """Generates a list of date strings between two dates."""
    dates = []
    delta = end_date - start_date
    for i in range(delta.days + 1):
        day = start_date + timedelta(days=i)
        dates.append(day.strftime('%Y-%m-%d'))
    return dates

# Define the start and end dates for the data you want
start_date = date(2025, 2, 20)
end_date = date(2025, 2, 27) # e.g., for one week

date_strings = get_date_range(start_date, end_date)

# Create the list of full URLs
base_url = "http://aisdata.ais.dk/aisdk-"
file_urls = [f"{base_url}{d}.zip" for d in date_strings]

print(f"Found {len(file_urls)} days to process.")
print(f"Using {NUM_CORES} CPU cores.")
print(f"Outputting all data to: {OUTPUT_PARQUET_ROOT}")

# --- 3. RUN THE PROCESSING IN PARALLEL ---
if __name__ == "__main__":
    # We need a simple wrapper function for the multiprocessing Pool,
    # because the 'out_path' argument is always the same.
    # 'partial' freezes the 'out_path' argument for our 'fn' function.
    
    process_day = partial(fn, out_path=OUTPUT_PARQUET_ROOT)

    # Create a processing pool and map the 'process_day' function
    # across the list of 'file_urls'.
    with multiprocessing.Pool(processes=NUM_CORES) as pool:
        # pool.map will run the function for each URL in the list
        # and automatically manage the parallel jobs.
        pool.map(process_day, file_urls)

    print("--- All preprocessing jobs complete! ---")
    print(f"Your training data is ready at: {OUTPUT_PARQUET_ROOT}")