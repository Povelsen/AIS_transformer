# --- run_2_train.py ---
import config
from pipeline import DataManager, Trainer
from model import VesselTransformer
import torch
import os

def main():
    print("="*60)
    print(f"STEP 2: TRAINING ON {config.DEVICE}")
    print("="*60)

    # 1. Prepare Data
    print("Loading Dataset...")
    data_manager = DataManager(
        parquet_root=config.PARQUET_OUT,
        test_size=0.15,
        val_size=0.15,
    )
    
    try:
        data_manager.create_data_splits()
    except FileNotFoundError:
        print("Error: Processed data not found. Run 'run_1_preprocessing.py' first.")
        return

    train_loader, val_loader, _ = data_manager.get_dataloaders(
        history_len=config.HISTORY_LEN,
        future_len=config.FUTURE_LEN,
        batch_size=config.BATCH_SIZE,
        num_workers=0 # Set to 0 for stability on MacOS, 2+ on Linux
    )

    if len(train_loader) == 0:
        print("Error: No valid tracks found. Check preprocessing logic.")
        return

    # 2. Initialize Model
    print(f"Initializing Transformer ({config.NUM_LAYERS} layers, {config.D_MODEL} dim)...")
    model = VesselTransformer(
        input_features=4, # Lat, Lon, SOG, COG
        d_model=config.D_MODEL,
        nhead=config.NHEAD,
        num_encoder_layers=config.NUM_LAYERS,
        num_decoder_layers=config.NUM_LAYERS,
        dim_feedforward=config.DIM_FEEDFORWARD,
        dropout=0.1
    )

    # 3. Train
    # Get normalization stats from loader to calculate error in meters
    norm_stats = (train_loader.dataset.mean, train_loader.dataset.std)
    
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=config.LEARNING_RATE,
        device=config.DEVICE,
        checkpoint_dir=config.CHECKPOINT_DIR,
        normalization_stats=norm_stats
    )

    trainer.train(config.NUM_EPOCHS)
    trainer.plot_loss() # Saves 'training_metrics.png'

    print("\nDone! Best model saved. You can now run 'run_3_evaluate.py'.")

if __name__ == "__main__":
    main()