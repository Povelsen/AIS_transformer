"""Entry point for running preprocessing and model training/evaluation."""

import torch

from preprocessing import AISDataPreprocessor
from model import VesselTransformer
from pipeline import DataManager, Trainer, Evaluator


# --- 1. GLOBAL CONFIGURATION ---
CONFIG = {
    # --- Step 1: Preprocessing ---
    "DATA_FOLDER": "test_data",  # Folder containing your CSV files
    "PARQUET_ROOT": "test_data/ais_parquet_data",  # Output location for processed data
    "NUM_CORES": 7,  # Number of CPU cores for parallel processing

    # --- Step 2: Training Pipeline ---
    "DEVICE": torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")),

    # Data params
    "HISTORY_LEN": 60,  # Use 60 points (~1 hr) to predict the 61st
    "FUTURE_LEN": 1,    # For sliding window (target is 1 step ahead)

    # Model Hyperparameters
    "D_MODEL": 128,     # Embedding dimension
    "NHEAD": 8,         # Number of attention heads
    "NUM_LAYERS": 4,    # Number of Transformer Encoder layers
    "DIM_FEEDFORWARD": 512,  # Hidden dim in feedforward network

    # Training Hyperparameters
    "NUM_EPOCHS": 1,
    "BATCH_SIZE": 32,
    "LEARNING_RATE": 0.0001,
    "NUM_WORKERS": 4,   # For DataLoader
}


# --- 2. DEFINE THE STEPS ---

def run_preprocessing(config: dict) -> None:
    """Runs the data ingestion and preprocessing step on local CSV files."""
    print("--- STEP 1: RUNNING PREPROCESSING ON LOCAL CSV FILES ---")
    preprocessor = AISDataPreprocessor(
        out_path=config["PARQUET_ROOT"],
        num_cores=config["NUM_CORES"],
    )
    preprocessor.process_local_csv(config["DATA_FOLDER"])
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

    # get the loaders
    train_loader, val_loader, test_files = data_manager.get_dataloaders(
        history_len=config["HISTORY_LEN"],
        future_len=config["FUTURE_LEN"],
        batch_size=config["BATCH_SIZE"],
        num_workers=config["NUM_WORKERS"],
    )

    # use train_loader to get normalization stats
    norm_stats = (train_loader.dataset.mean, train_loader.dataset.std)

    # 2. Initialize Model
    model = VesselTransformer(
        input_features=4,  # Lat, Lon, SOG, COG
        d_model=config["D_MODEL"],
        nhead=config["NHEAD"],
        num_encoder_layers=config["NUM_LAYERS"],
        num_decoder_layers=config["NUM_LAYERS"],
        dim_feedforward=config["DIM_FEEDFORWARD"],
    )

    # 3. Train Model
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=config["LEARNING_RATE"],
        device=config["DEVICE"],
        normalization_stats=norm_stats,
    )

    trained_model = trainer.train(config["NUM_EPOCHS"])
    trainer.plot_loss()  # Saves loss_plot.png

    # 4. Evaluate Model
    evaluator = Evaluator(
        trained_model,
        device=config["DEVICE"],
        normalization_stats=norm_stats,
    )
    evaluator.evaluate_and_plot(test_files, history_len=config["HISTORY_LEN"])

    print("--- STEP 2: TRAINING PIPELINE COMPLETE ---")
    print("Check 'loss_plot.png' and 'prediction_map.html' for results.")



# --- 3. RUN THE PROJECT ---

if __name__ == "__main__":
    # ---
    # TO RUN:
    # 1. Place your CSV files in the 'data' folder.
    # 2. Uncomment 'run_preprocessing(CONFIG)' and run the file.
    # 3. Once complete, comment 'run_preprocessing(CONFIG)' again.
    # 4. Uncomment 'run_training_pipeline(CONFIG)' and run the file.
    # ---

    #run_preprocessing(CONFIG)  # Run this first to process CSV files

    run_training_pipeline(CONFIG)  # Run this after preprocessing is done