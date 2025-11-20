import os
from typing import List, Sequence, Tuple, Optional
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
import torch.nn as nn
from haversine import Unit, haversine
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# --- Helper utilities ---
def haversine_distance(lat1, lon1, lat2, lon2):
    return haversine((lat1, lon1), (lat2, lon2), unit=Unit.METERS)

def circular_loss(pred_cog, true_cog):
    diff = (pred_cog - true_cog + 180) % 360 - 180
    return (diff ** 2).mean()

# --- Data management (Same as before, just abbreviated for brevity) ---
class DataManager:
    def __init__(self, parquet_root, test_size=0.15, val_size=0.15):
        self.parquet_root = parquet_root
        self.test_size = test_size
        self.val_size = val_size
        self.segment_files = []
        self.train_files = []
        self.val_files = []
        self.test_files = []

    def _find_segment_files(self):
        segment_paths = []
        if not os.path.exists(self.parquet_root): return
        for mmsi_dir in os.listdir(self.parquet_root):
            if "MMSI=" not in mmsi_dir: continue
            m_path = os.path.join(self.parquet_root, mmsi_dir)
            for seg_dir in os.listdir(m_path):
                if "Segment=" not in seg_dir: continue
                s_path = os.path.join(m_path, seg_dir)
                for f in os.listdir(s_path):
                    if f.endswith(".parquet"):
                        segment_paths.append(os.path.join(s_path, f))
        self.segment_files = segment_paths

    def create_data_splits(self):
        self._find_segment_files()
        if not self.segment_files: raise FileNotFoundError("No Parquet files found.")
        train_val, self.test_files = train_test_split(self.segment_files, test_size=self.test_size, random_state=42)
        self.train_files, self.val_files = train_test_split(train_val, test_size=self.val_size/(1-self.test_size), random_state=42)
        print(f"Split: {len(self.train_files)} Train, {len(self.val_files)} Val, {len(self.test_files)} Test")

    def get_dataloaders(self, history_len, future_len, batch_size, num_workers=0):
        train_dataset = VesselTrajectoryDataset(self.train_files, history_len, future_len)
        val_dataset = VesselTrajectoryDataset(self.val_files, history_len, future_len)
        val_dataset.mean = train_dataset.mean
        val_dataset.std = train_dataset.std
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        return train_loader, val_loader, self.test_files

class VesselTrajectoryDataset(Dataset):
    def __init__(self, segment_files, history_len, future_len, stride=1):
        self.segment_files = segment_files
        self.history_len = history_len
        self.future_len = future_len
        self.window_len = history_len + future_len
        self.features = ["Latitude", "Longitude", "SOG", "COG"]
        self.samples = []
        self._cache = {}
        self.mean = None
        self.std = None
        
        # Indexing
        for f in self.segment_files:
            try:
                meta = pq.read_metadata(f)
                if meta.num_rows >= self.window_len:
                    for i in range(0, meta.num_rows - self.window_len + 1, stride):
                        self.samples.append((f, i))
            except: continue
            
        if self.samples: self._compute_stats()

    def _compute_stats(self):
        # Calculate stats on a subset for speed
        subset = self.segment_files[:min(200, len(self.segment_files))]
        data_list = []
        for f in subset:
            try:
                t = pq.read_table(f, columns=self.features)
                arr = np.column_stack([t.column(c).to_numpy().astype(np.float32) for c in self.features])
                data_list.append(arr)
            except: continue
        all_data = np.vstack(data_list)
        self.mean = torch.tensor(all_data.mean(axis=0), dtype=torch.float32)
        self.std = torch.tensor(all_data.std(axis=0), dtype=torch.float32).clamp(min=1e-6)

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        f, start = self.samples[idx]
        if f not in self._cache:
            t = pq.read_table(f, columns=self.features)
            arr = np.column_stack([t.column(c).to_numpy().astype(np.float32) for c in self.features])
            self._cache[f] = torch.from_numpy(arr)
        
        data = self._cache[f][start : start + self.window_len]
        hist = (data[:self.history_len] - self.mean) / self.std
        fut = (data[self.history_len:] - self.mean) / self.std
        return hist, fut

# --- Training with detailed Metrics ---
class Trainer:
    def __init__(self, model, train_loader, val_loader, learning_rate, device, normalization_stats, checkpoint_dir=None):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.norm_mean = normalization_stats[0].to(device)
        self.norm_std = normalization_stats[1].to(device)
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.mse = nn.MSELoss()
        self.history = {"train_loss": [], "val_loss": [], "val_dist_error": []}
        
        # --- NEW: Handle Checkpointing ---
        self.checkpoint_dir = checkpoint_dir
        if self.checkpoint_dir:
            os.makedirs(self.checkpoint_dir, exist_ok=True)

    def _combined_loss(self, pred, true):
        # Simple MSE on normalized values
        mse_pos = self.mse(pred[:,:,:2], true[:,:,:2]) # Lat/Lon
        mse_speed = self.mse(pred[:,:,2], true[:,:,2]) # Speed
        return mse_pos + mse_speed

    def _calculate_meters_error(self, pred_norm, true_norm):
        """Converts normalized batch back to Lat/Lon and calculates distance error."""
        # Denormalize
        pred = pred_norm * self.norm_std + self.norm_mean
        true = true_norm * self.norm_std + self.norm_mean
        
        # Extract Lat/Lon (Batch, Seq, Features)
        p_lat, p_lon = pred[:, :, 0], pred[:, :, 1]
        t_lat, t_lon = true[:, :, 0], true[:, :, 1]
        
        # Haversine approximation for batch (simplified for performance)
        R = 6371000  # Earth radius in meters
        phi1, phi2 = torch.deg2rad(p_lat), torch.deg2rad(t_lat)
        dphi = torch.deg2rad(t_lat - p_lat)
        dlambda = torch.deg2rad(t_lon - p_lon)
        
        a = torch.sin(dphi/2)**2 + torch.cos(phi1)*torch.cos(phi2)*torch.sin(dlambda/2)**2
        c = 2 * torch.atan2(torch.sqrt(a), torch.sqrt(1-a))
        dist = R * c
        return dist.mean().item()

    def train(self, epochs):
        print(f"\nStarting Training for {epochs} epochs...")
        best_val_loss = float('inf') # Track best loss

        for epoch in range(epochs):
            self.model.train()
            avg_loss = 0
            for x, y in tqdm(self.train_loader, desc=f"Epoch {epoch+1} [Train]", leave=False):
                x, y = x.to(self.device), y.to(self.device)
                
                # Teacher forcing input: Start token + Target shifted right
                start_token = x[:, -1:, :] 
                dec_input = start_token if y.size(1) == 1 else torch.cat([start_token, y[:, :-1, :]], dim=1)
                
                pred = self.model(x, dec_input)
                loss = self._combined_loss(pred, y)
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                avg_loss += loss.item()
            
            train_loss = avg_loss / len(self.train_loader)
            self.history["train_loss"].append(train_loss)
            
            # Validation
            self.model.eval()
            val_loss = 0
            dist_error = 0
            with torch.no_grad():
                for x, y in self.val_loader:
                    x, y = x.to(self.device), y.to(self.device)
                    start_token = x[:, -1:, :]
                    dec_input = start_token if y.size(1) == 1 else torch.cat([start_token, y[:, :-1, :]], dim=1)
                    
                    pred = self.model(x, dec_input)
                    val_loss += self._combined_loss(pred, y).item()
                    dist_error += self._calculate_meters_error(pred, y)
            
            val_loss /= len(self.val_loader)
            dist_error /= len(self.val_loader)
            self.history["val_loss"].append(val_loss)
            self.history["val_dist_error"].append(dist_error)
            
            print(f"Epoch {epoch+1} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Error: {dist_error:.2f} meters/step")
            
            # --- NEW: Save Best Model ---
            if self.checkpoint_dir and val_loss < best_val_loss:
                best_val_loss = val_loss
                save_path = os.path.join(self.checkpoint_dir, "best_model.pth")
                torch.save(self.model.state_dict(), save_path)
        
        return self.model

    def plot_loss(self):
        # (Keep your existing plot_loss code here, no changes needed)
        fig, ax1 = plt.subplots(figsize=(10, 5))
        
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss (MSE)', color='tab:blue')
        ax1.plot(self.history['train_loss'], label='Train Loss', color='tab:blue')
        ax1.plot(self.history['val_loss'], label='Val Loss', color='tab:blue', linestyle='--')
        ax1.tick_params(axis='y', labelcolor='tab:blue')
        
        ax2 = ax1.twinx()
        ax2.set_ylabel('Distance Error (Meters)', color='tab:red')
        ax2.plot(self.history['val_dist_error'], label='Val Error (m)', color='tab:red')
        ax2.tick_params(axis='y', labelcolor='tab:red')
        
        plt.title("Training Progress: Loss vs Physical Error")
        fig.tight_layout()
        plt.savefig("training_metrics.png")
        print("Saved training_metrics.png")

# --- Evaluation with Static Plotting ---
class Evaluator:
    def __init__(self, model, device, normalization_stats):
        self.model = model.to(device)
        self.device = device
        self.norm_mean = normalization_stats[0].to(device)
        self.norm_std = normalization_stats[1].to(device)

    def evaluate_and_plot(self, test_files, history_len, file_idx=0):
        if not test_files: return
        file_path = test_files[file_idx]
        
        # Load Data
        df = pd.read_parquet(file_path, columns=["Latitude", "Longitude", "SOG", "COG", "Timestamp"])
        df["Timestamp"] = pd.to_datetime(df["Timestamp"])
        
        # Prepare Input (First 'history_len' points)
        if len(df) < history_len + 10:
            print("Track too short for evaluation.")
            return
            
        input_data = df.iloc[:history_len][["Latitude", "Longitude", "SOG", "COG"]].values.astype(np.float32)
        
        # Normalize
        mean = self.norm_mean.cpu().numpy()
        std = self.norm_std.cpu().numpy()
        norm_input = (input_data - mean) / std
        
        src = torch.tensor(norm_input).unsqueeze(0).to(self.device) # [1, H, 4]
        
        # Autoregressive Prediction (Predict next 60 minutes)
        # Assuming data is approx 1 min resolution, predict 60 steps
        steps_to_predict = 60 
        decoder_input = src[:, -1:, :] # Start with last known point
        
        predictions = []
        self.model.eval()
        with torch.no_grad():
            for _ in range(steps_to_predict):
                out = self.model(src, decoder_input) # [1, t, 4]
                next_step = out[:, -1:, :] # [1, 1, 4]
                predictions.append(next_step.cpu().numpy())
                decoder_input = torch.cat([decoder_input, next_step], dim=1)
        
        # Denormalize predictions
        preds = np.concatenate(predictions, axis=1).squeeze(0) # [steps, 4]
        preds = preds * std + mean
        
        # Create DataFrames for plotting
        pred_df = pd.DataFrame(preds, columns=["Latitude", "Longitude", "SOG", "COG"])
        true_df = df.iloc[:history_len + steps_to_predict] # Full ground truth
        
        # Calculate accuracy of this specific run
        last_pred_lat, last_pred_lon = pred_df.iloc[-1]["Latitude"], pred_df.iloc[-1]["Longitude"]
        last_true_lat, last_true_lon = true_df.iloc[-1]["Latitude"], true_df.iloc[-1]["Longitude"]
        final_error = haversine_distance(last_pred_lat, last_pred_lon, last_true_lat, last_true_lon)
        
        print(f"Final predicted point is {final_error:.2f} meters away from actual location.")

        self._plot_static(true_df, pred_df, history_len)

    def _plot_static(self, true_df, pred_df, split_index):
        plt.figure(figsize=(10, 8))
        
        # Plot History (Black)
        plt.plot(true_df.iloc[:split_index]["Longitude"], true_df.iloc[:split_index]["Latitude"], 
                 'k-', linewidth=2, label='History (Context)')
        
        # Plot True Future (Green)
        plt.plot(true_df.iloc[split_index:]["Longitude"], true_df.iloc[split_index:]["Latitude"], 
                 'g--', linewidth=2, label='Actual Future')
        
        # Plot Predicted Future (Red)
        plt.plot(pred_df["Longitude"], pred_df["Latitude"], 
                 'r-o', markersize=3, label='Predicted Future')
        
        plt.legend()
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.title("Vessel Trajectory Prediction (Static Plot)")
        plt.grid(True, alpha=0.3)
        
        # Ensure aspect ratio is decent for maps
        plt.axis('equal')
        
        plt.savefig("prediction_static.png", dpi=150)
        print("Saved prediction plot to 'prediction_static.png'")