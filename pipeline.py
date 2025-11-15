import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pyarrow.parquet as pq
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import folium
from sklearn.model_selection import train_test_split
from haversine import haversine, Unit

# --- Helper Function ---

def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate distance between two points in meters."""
    return haversine((lat1, lon1), (lat2, lon2), unit=Unit.METERS)

# --- Data Management Classes ---

class DataManager:
    """
    Handles finding, splitting, and loading trajectory data into DataLoaders.
    """
    def __init__(self, parquet_root, test_size=0.15, val_size=0.15, random_state=42):
        self.parquet_root = parquet_root
        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state
        self.segment_files = []
        self.train_files = []
        self.val_files = []
        self.test_files = []

    def _find_segment_files(self):
        """Finds all (MMSI, Segment) pairs from the partitioned directory."""
        print("Finding segment files...")
        segment_paths = []
        for mmsi_dir in os.listdir(self.parquet_root):
            if mmsi_dir.startswith("MMSI="):
                mmsi_path = os.path.join(self.parquet_root, mmsi_dir)
                for segment_dir in os.listdir(mmsi_path):
                    if segment_dir.startswith("Segment="):
                        segment_path = os.path.join(mmsi_path, segment_dir)
                        for f in os.listdir(segment_path):
                            if f.endswith(".parquet"):
                                segment_paths.append(os.path.join(segment_path, f))
                                break
        self.segment_files = segment_paths
        print(f"Found {len(self.segment_files)} total segments.")

    def create_data_splits(self):
        """Splits the data at the trajectory (segment) level."""
        self._find_segment_files()
        
        if not self.segment_files:
            raise FileNotFoundError(f"No parquet files found at {self.parquet_root}. Did preprocessing run correctly?")

        # Split into train and (val + test)
        train_val_files, self.test_files = train_test_split(
            self.segment_files, 
            test_size=self.test_size, 
            random_state=self.random_state
        )
        
        # Split train_val into train and val
        self.train_files, self.val_files = train_test_split(
            train_val_files,
            test_size=(self.val_size / (1.0 - self.test_size)), # Correct splitting
            random_state=self.random_state
        )
        
        print(f"Train segments: {len(self.train_files)}")
        print(f"Val segments: {len(self.val_files)}")
        print(f"Test segments: {len(self.test_files)}")

    def get_dataloaders(self, history_len, future_len, batch_size, num_workers=4):
        """Creates and returns the PyTorch DataLoaders."""
        train_dataset = VesselTrajectoryDataset(self.train_files, history_len, future_len)
        val_dataset = VesselTrajectoryDataset(self.val_files, history_len, future_len)
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=num_workers
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=num_workers
        )
        
        return train_loader, val_loader, self.test_files


class VesselTrajectoryDataset(Dataset):
    """
    PyTorch Dataset to create (history, target) windows from trajectory files.
    """
    def __init__(self, segment_files, history_len, future_len):
        self.segment_files = segment_files
        self.history_len = history_len
        self.future_len = future_len
        self.window_len = history_len + future_len
        
        self.features = ['Latitude', 'Longitude', 'SOG', 'COG']
        self.targets = ['Latitude', 'Longitude', 'SOG', 'COG']
        
        self.samples = []
        print(f"Preprocessing {len(segment_files)} segments for Dataset...")
        for file_path in self.segment_files:
            try:
                # Read metadata to get length without loading the whole file
                num_rows = pq.read_metadata(file_path).num_rows
                if num_rows >= self.window_len:
                    # This segment is long enough, add all possible windows
                    for i in range(num_rows - self.window_len + 1):
                        self.samples.append((file_path, i))
            except Exception as e:
                print(f"Warning: Could not read metadata for {file_path}: {e}")
        
        print(f"Created {len(self.samples)} total windows.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file_path, start_idx = self.samples[idx]
        
        # Load the single trajectory segment
        df = pd.read_parquet(file_path, columns=self.features)
        
        # Get the window
        window_df = df.iloc[start_idx : start_idx + self.window_len]
        
        # x: The first 'history_len' points (inputs)
        x = window_df[self.features].iloc[:self.history_len].values
        
        # y: The *next* 'history_len' points (targets), shifted by 1
        # This creates the "predict the next step" behavior
        y = window_df[self.targets].iloc[1:self.history_len + 1].values

        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


# --- Training and Evaluation Classes ---

class Trainer:
    """
    Handles the model training and validation loops.
    """
    def __init__(self, model, train_loader, val_loader, learning_rate, device):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.criterion = nn.MSELoss() # Mean Squared Error is good for coordinate data
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.history = {'train_loss': [], 'val_loss': []}

    def train(self, num_epochs):
        """Runs the training and validation loops for a given number of epochs."""
        print("\n--- Starting Training ---")
        for epoch in range(num_epochs):
            self.model.train()
            running_train_loss = 0.0
            for x_batch, y_batch in self.train_loader:
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                
                # Forward pass
                predictions = self.model(x_batch)
                loss = self.criterion(predictions, y_batch)
                
                # Backward pass and optimize
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                running_train_loss += loss.item()
            
            epoch_train_loss = running_train_loss / len(self.train_loader)
            self.history['train_loss'].append(epoch_train_loss)
            
            # --- Validation ---
            self.model.eval()
            running_val_loss = 0.0
            with torch.no_grad():
                for x_batch, y_batch in self.val_loader:
                    x_batch = x_batch.to(self.device)
                    y_batch = y_batch.to(self.device)
                    
                    predictions = self.model(x_batch)
                    loss = self.criterion(predictions, y_batch)
                    running_val_loss += loss.item()
            
            epoch_val_loss = running_val_loss / len(self.val_loader)
            self.history['val_loss'].append(epoch_val_loss)
            
            print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {epoch_train_loss:.6f} | Val Loss: {epoch_val_loss:.6f}")
        
        print("--- Training Complete ---")
        return self.model

    def plot_loss(self):
        """Plots the training and validation loss curves."""
        plt.figure(figsize=(10, 5))
        plt.plot(self.history['train_loss'], label='Training Loss')
        plt.plot(self.history['val_loss'], label='Validation Loss')
        plt.title('Training and Validation Loss Over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Loss (MSE)')
        plt.legend()
        plt.grid(True)
        plt.savefig("loss_plot.png")
        print("Loss plot saved to loss_plot.png")


class Evaluator:
    """
    Handles autoregressive evaluation, metric calculation, and plotting.
    """
    def __init__(self, model, device):
        self.model = model.to(device)
        self.device = device

    def _perform_prediction(self, prompt_df, num_to_predict, history_len):
        """Helper to run the autoregressive prediction loop."""
        self.model.eval()
        
        model_input = torch.tensor(
            prompt_df[['Latitude', 'Longitude', 'SOG', 'COG']].values,
            dtype=torch.float32
        ).unsqueeze(0).to(self.device)
        
        predicted_features = []
        with torch.no_grad():
            for _ in range(num_to_predict):
                prediction = self.model(model_input)
                
                # Get the *very last* prediction (all 4 features)
                next_input_vec_tensor = prediction[0, -1, :] 
                predicted_features.append(next_input_vec_tensor.cpu().numpy())
                
                # Reshape for next input
                next_input_reshaped = next_input_vec_tensor.unsqueeze(0).unsqueeze(0)
                
                # Append new prediction, discard oldest point
                model_input = torch.cat([model_input[:, 1:, :], next_input_reshaped], dim=1)
        
        return pd.DataFrame(
            predicted_features, 
            columns=['Pred_Lat', 'Pred_Lon', 'Pred_SOG', 'Pred_COG']
        )

    def _calculate_metrics(self, true_df, pred_df, t_prompt_end):
        """Helper to calculate the error table."""
        results_df = pd.concat([true_df.reset_index(drop=True), pred_df.reset_index(drop=True)], axis=1)
        
        try:
            t_20m_idx = (true_df['Timestamp'] >= (t_prompt_end + pd.Timedelta(minutes=20))).idxmax()
        except ValueError: t_20m_idx = -1
        
        try:
            t_1h_idx = (true_df['Timestamp'] >= (t_prompt_end + pd.Timedelta(hours=1))).idxmax()
        except ValueError: t_1h_idx = -1
        
        t_last_idx = len(true_df) - 1
        
        indices_to_check = {"20 Minutes": t_20m_idx, "1 Hour": t_1h_idx, "Last Point": t_last_idx}
        error_table = []

        for name, idx in indices_to_check.items():
            if 0 <= idx < len(results_df):
                row = results_df.iloc[idx]
                error = haversine_distance(
                    row['Latitude'], row['Longitude'],
                    row['Pred_Lat'], row['Pred_Lon']
                )
                error_table.append({
                    "Time Horizon": name,
                    "Error (meters)": f"{error:.2f}"
                })
        
        return pd.DataFrame(error_table)

    def _plot_map(self, full_true_df, pred_df, start_pred_point):
        """Helper to generate and save the Folium map."""
        print("\nGenerating Folium map...")
        m = folium.Map()
        
        true_points = list(zip(full_true_df['Latitude'], full_true_df['Longitude']))
        pred_points = list(zip(pred_df['Pred_Lat'], pred_df['Pred_Lon']))
        
        folium.PolyLine(true_points, color='blue', weight=4, opacity=0.8, popup='True Path').add_to(m)
        folium.PolyLine(pred_points, color='red', weight=4, opacity=0.8, popup='Predicted Path').add_to(m)
        folium.Marker(
            location=start_pred_point,
            popup='Start of Prediction',
            icon=folium.Icon(color='green')
        ).add_to(m)
        
        m.fit_bounds(m.get_bounds())
        map_path = "prediction_map.html"
        m.save(map_path)
        print(f"Map saved to {map_path}")

    def evaluate_and_plot(self, test_files, history_len, file_to_plot=0):
        """
        Runs the full evaluation on a single test file and plots the result.
        """
        if not test_files:
            print("No test files available for evaluation.")
            return

        test_file = test_files[file_to_plot]
        print(f"\n--- Evaluating on Test File: {test_file} ---")
        df = pd.read_parquet(test_file, columns=['Timestamp', 'Latitude', 'Longitude', 'SOG', 'COG'])
        
        df["Timestamp"] = pd.to_datetime(df["Timestamp"])
        t_start = df['Timestamp'].iloc[0]
        t_prompt_end = t_start + pd.Timedelta(hours=1)
        
        prompt_df = df[df['Timestamp'] <= t_prompt_end]
        prompt_idx = len(prompt_df)
        
        if prompt_idx > history_len:
             prompt_idx = history_len
             prompt_df = prompt_df.iloc[:prompt_idx]
        elif prompt_idx < 10: 
             print(f"Skipping file {test_file}, too few prompt points.")
             return

        print(f"Using first {prompt_idx} data points as prompt.")
        
        true_df = df.iloc[prompt_idx:]
        num_to_predict = len(true_df)
        if num_to_predict == 0:
             print(f"Skipping file {test_file}, no future data to predict.")
             return

        pred_df = self._perform_prediction(prompt_df, num_to_predict, history_len)
        
        error_table_df = self._calculate_metrics(true_df, pred_df, t_prompt_end)
        print("\n--- Prediction Error Table ---")
        print(error_table_df.to_string(index=False))
        
        start_pred_point_coords = (true_df.iloc[0]['Latitude'], true_df.iloc[0]['Longitude'])
        self._plot_map(df, pred_df, start_pred_point_coords)