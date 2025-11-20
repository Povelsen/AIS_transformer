# --- config.py ---
import torch
import os

# 1. PATHS
# Get current directory to avoid hardcoded paths
BASE_DIR = os.getcwd()
DATA_RAW = os.path.join(BASE_DIR, "raw_ais_data")           # Where zip/csv files are
PARQUET_OUT = os.path.join(BASE_DIR, "cleaned_test_data/parquet") # Where cleaned data goes
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")    # Where model weights are saved

# 2. COMPUTE
# Auto-detect GPU (CUDA for Nvidia, MPS for Mac, CPU otherwise)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
NUM_CORES = 4  # For data processing

# 3. DATA PARAMETERS
HISTORY_LEN = 60   # How many past points to look at
FUTURE_LEN = 1     # Predict 1 step ahead (Autoregressive)
SHIP_TYPE = "Cargo" # Filter for this ship type

# 4. MODEL HYPERPARAMETERS
D_MODEL = 128        # Size of the vector representation
NHEAD = 4            # Attention heads
NUM_LAYERS = 3       # Transformer layers
DIM_FEEDFORWARD = 256 

# 5. TRAINING PARAMETERS
NUM_EPOCHS = 2
BATCH_SIZE = 64
LEARNING_RATE = 0.001