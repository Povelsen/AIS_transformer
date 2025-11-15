"""Entry point for running preprocessing and model training/evaluation."""

from datetime import date
import torch

from preprocessing import AISDataPreprocessor
from model import VesselTransformer
from pipeline import DataManager, Trainer, Evaluator


# --- 1. GLOBAL CONFIGURATION ---
CONFIG = {
    # --- Step 1: Preprocessing ---
    "PARQUET_ROOT": "data/ais_parquet_data",  # IMPORTANT: Change this path
    "START_DATE": date(2025, 2, 20),
    "END_DATE": date(2025, 2, 27),
    "NUM_CORES": 7,  # Number of CPU cores for parallel processing

    # --- Step 2: Training Pipeline ---
    "DEVICE": torch.device("cuda" if torch.cuda.is_available() else "cpu"),

    # Data params
    "HISTORY_LEN": 60,  # Use 60 points (~1 hr) to predict the 61st
    "FUTURE_LEN": 1,    # For sliding window (target is 1 step ahead)

    # Model Hyperparameters
    "D_MODEL": 128,     # Embedding dimension
    "NHEAD": 8,         # Number of attention heads
    "NUM_LAYERS": 4,    # Number of Transformer Encoder layers
    "DIM_FEEDFORWARD": 512,  # Hidden dim in feedforward network

    # Training Hyperparameters
    "NUM_EPOCHS": 10,
    "BATCH_SIZE": 32,
    "LEARNING_RATE": 0.0001,
    "NUM_WORKERS": 4,   # For DataLoader
}


# --- 2. DEFINE THE STEPS ---

def run_preprocessing(config: dict) -> None:
    """Runs the data ingestion and preprocessing step."""
    print("--- STEP 1: RUNNING PREPROCESSING ---")
    preprocessor = AISDataPreprocessor(
        out_path=config["PARQUET_ROOT"],
        num_cores=config["NUM_CORES"],
    )
    preprocessor.process_date_range_parallel(
        start_date=config["START_DATE"],
        end_date=config["END_DATE"],
    )
    print("--- STEP 1: PREPROCESSING COMPLETE ---")


def run_training_pipeline(config: dict) -> None:
    """Runs the full model training and evaluation pipeline."""
    print("--- STEP 2: RUNNING TRAINING PIPELINE ---")

    # 1. Load Data
    data_manager = DataManager(
        parquet_root=config["PARQUET_ROOT"],
        test_size=0.15,
        val_size=0.15,
    )
    data_manager.create_data_splits()
    train_loader, val_loader, test_files = data_manager.get_dataloaders(
        history_len=config["HISTORY_LEN"],
        future_len=config["FUTURE_LEN"],
        batch_size=config["BATCH_SIZE"],
        num_workers=config["NUM_WORKERS"],
    )

    # 2. Initialize Model
    model = VesselTransformer(
        input_features=4,  # Lat, Lon, SOG, COG
        d_model=config["D_MODEL"],
        nhead=config["NHEAD"],
        num_layers=config["NUM_LAYERS"],
        dim_feedforward=config["DIM_FEEDFORWARD"],
    )

    # 3. Train Model
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=config["LEARNING_RATE"],
        device=config["DEVICE"],
    )

    trained_model = trainer.train(config["NUM_EPOCHS"])
    trainer.plot_loss()  # Saves loss_plot.png

    # 4. Evaluate Model
    evaluator = Evaluator(trained_model, device=config["DEVICE"])
    evaluator.evaluate_and_plot(test_files, history_len=config["HISTORY_LEN"])

    print("--- STEP 2: TRAINING PIPELINE COMPLETE ---")
    print("Check 'loss_plot.png' and 'prediction_map.html' for results.")


# --- 3. RUN THE PROJECT ---

if __name__ == "__main__":
    # ---
    # TO RUN:
    # 1. Make sure PARQUET_ROOT is set correctly.
    # 2. Uncomment 'run_preprocessing(CONFIG)' and run the file.
    # 3. Once complete, comment 'run_preprocessing(CONFIG)' again.
    # 4. Uncomment 'run_training_pipeline(CONFIG)' and run the file.
    # ---

    # run_preprocessing(CONFIG)

    run_training_pipeline(CONFIG)
