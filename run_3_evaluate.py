# --- run_3_evaluate.py ---
import config
from pipeline import DataManager, Evaluator
from model import VesselTransformer
import torch
import os
import random

def main():
    print("="*60)
    print("STEP 3: EVALUATION & PLOTTING")
    print("="*60)

    checkpoint_path = os.path.join(config.CHECKPOINT_DIR, "best_model.pt")
    if not os.path.exists(checkpoint_path):
        print(f"Error: No model found at {checkpoint_path}")
        print("Run 'run_2_train.py' first.")
        return

    # 1. Re-initialize Model Structure
    model = VesselTransformer(
        input_features=4,
        d_model=config.D_MODEL,
        nhead=config.NHEAD,
        num_encoder_layers=config.NUM_LAYERS,
        num_decoder_layers=config.NUM_LAYERS,
        dim_feedforward=config.DIM_FEEDFORWARD,
    )

    # 2. Load Weights & Stats
    print(f"Loading model from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=config.DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Retrieve normalization stats saved during training
    # (This ensures we decode the data exactly how it was encoded)
    if 'norm_mean' not in checkpoint:
        print("Error: Old checkpoint format. Please re-train with 'run_2_train.py'.")
        return
        
    norm_stats = (checkpoint['norm_mean'], checkpoint['norm_std'])

    # 3. Get Test Files
    data_manager = DataManager(parquet_root=config.PARQUET_OUT)
    data_manager.create_data_splits()
    test_files = data_manager.test_files

    if not test_files:
        print("No test files found.")
        return

    # 4. Run Evaluation on a random file
    evaluator = Evaluator(
        model=model,
        device=config.DEVICE,
        normalization_stats=norm_stats
    )

    # Pick a random file from test set to see different results every time
    rand_idx = random.randint(0, len(test_files) - 1)
    print(f"Evaluating File #{rand_idx}: {test_files[rand_idx]}")
    
    evaluator.evaluate_and_plot(
        test_files=test_files, 
        history_len=config.HISTORY_LEN, 
        file_idx=rand_idx
    )

    print("\nDone! Check 'prediction_static.png' to see the result.")

if __name__ == "__main__":
    main()