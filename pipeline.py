"""Data loading, training, and evaluation helpers for the AIS Transformer."""

from __future__ import annotations

import os
from typing import List, Sequence, Tuple

import folium
import matplotlib.pyplot as plt
import pandas as pd
import pyarrow.parquet as pq
import torch
import torch.nn as nn
from haversine import Unit, haversine
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


# --- Helper utilities -------------------------------------------------------------------------

def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Return the distance between two lat/lon points in meters."""

    return haversine((lat1, lon1), (lat2, lon2), unit=Unit.METERS)


# --- Data management ---------------------------------------------------------------------------


class DataManager:
    """Finds preprocessed Parquet segments and builds DataLoaders."""

    def __init__(
        self,
        parquet_root: str,
        test_size: float = 0.15,
        val_size: float = 0.15,
        random_state: int = 42,
    ) -> None:
        self.parquet_root = parquet_root
        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state

        self.segment_files: List[str] = []
        self.train_files: List[str] = []
        self.val_files: List[str] = []
        self.test_files: List[str] = []

    # Internal helpers ------------------------------------------------------------------

    def _find_segment_files(self) -> None:
        """Populate ``self.segment_files`` with one parquet per (MMSI, Segment)."""

        if not os.path.isdir(self.parquet_root):
            raise FileNotFoundError(
                f"Parquet root '{self.parquet_root}' does not exist. "
                "Run preprocessing first."
            )

        print("Finding segment files...")
        segment_paths: List[str] = []
        for mmsi_dir in os.listdir(self.parquet_root):
            if not mmsi_dir.startswith("MMSI="):
                continue
            mmsi_path = os.path.join(self.parquet_root, mmsi_dir)
            if not os.path.isdir(mmsi_path):
                continue
            for segment_dir in os.listdir(mmsi_path):
                if not segment_dir.startswith("Segment="):
                    continue
                segment_path = os.path.join(mmsi_path, segment_dir)
                if not os.path.isdir(segment_path):
                    continue
                for file_name in os.listdir(segment_path):
                    if file_name.endswith(".parquet"):
                        segment_paths.append(os.path.join(segment_path, file_name))
                        break

        self.segment_files = segment_paths
        print(f"Found {len(self.segment_files)} total segments.")

    # Public API ------------------------------------------------------------------------

    def create_data_splits(self) -> None:
        """Split available segments into train/validation/test file lists."""

        self._find_segment_files()
        if not self.segment_files:
            raise FileNotFoundError(
                f"No parquet files found at {self.parquet_root}. Did preprocessing run correctly?"
            )

        train_val_files, self.test_files = train_test_split(
            self.segment_files,
            test_size=self.test_size,
            random_state=self.random_state,
        )
        self.train_files, self.val_files = train_test_split(
            train_val_files,
            test_size=self.val_size / (1.0 - self.test_size),
            random_state=self.random_state,
        )

        print(f"Train segments: {len(self.train_files)}")
        print(f"Val segments: {len(self.val_files)}")
        print(f"Test segments: {len(self.test_files)}")

    def get_dataloaders(
        self,
        history_len: int,
        future_len: int,
        batch_size: int,
        num_workers: int = 4,
        stride: int = 5
    ) -> Tuple[DataLoader, DataLoader, List[str]]:
        """Create PyTorch DataLoaders and return them alongside the test file list."""

        train_dataset = VesselTrajectoryDataset(self.train_files, history_len, future_len, stride=stride)
        val_dataset = VesselTrajectoryDataset(self.val_files, history_len, future_len, stride=stride)

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )
        return train_loader, val_loader, self.test_files


class VesselTrajectoryDataset(Dataset):
    def __init__(self, segment_files: Sequence[str], history_len: int, future_len: int, stride: int = 5) -> None:
        self.segment_files = list(segment_files)
        self.history_len = history_len
        self.future_len = future_len
        self.window_len = history_len + future_len
        self.stride = stride

        self.features = ["Latitude", "Longitude", "SOG", "COG"]
        self.targets = ["Latitude", "Longitude", "SOG", "COG"]

        self.samples: List[Tuple[str, int]] = []
        self._cache: dict[str, pd.DataFrame] = {}
        print(f"Preprocessing {len(self.segment_files)} segments for Dataset...")
        for file_path in self.segment_files:
            try:
                num_rows = pq.read_metadata(file_path).num_rows
            except Exception as exc:
                print(f"Warning: Could not read metadata for {file_path}: {exc}")
                continue

            if num_rows < self.window_len:
                continue

            for start_idx in range(0, num_rows - self.window_len + 1, self.stride):
                self.samples.append((file_path, start_idx))

        print(f"Created {len(self.samples)} total windows.")


    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        file_path, start_idx = self.samples[idx]

        # Lazy-load and cache the parquet for this file_path
        if file_path not in self._cache:
            # You only load each segment file once per Dataset lifetime
            self._cache[file_path] = pd.read_parquet(file_path, columns=self.features)

        df = self._cache[file_path]
        window_df = df.iloc[start_idx : start_idx + self.window_len]

        x = window_df[self.features].iloc[: self.history_len].values
        y = window_df[self.targets].iloc[1 : self.history_len + 1].values

        return (
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32),
        )


# --- Training ------------------------------------------------------------------------


class Trainer:
    """Handles the model training loop with checkpointing and live plotting."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        learning_rate: float,
        device: torch.device,
        checkpoint_dir: str = "checkpoints",
    ) -> None:
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.history = {"train_loss": [], "val_loss": []}
        self.checkpoint_dir = checkpoint_dir
        self.best_val_loss = float('inf')
        
        # Create checkpoint directory
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # Print model statistics
        self._print_model_stats()

    def _print_model_stats(self) -> None:
        """Print detailed model statistics."""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        print("\n" + "="*70)
        print("MODEL STATISTICS")
        print("="*70)
        print(f"Total Parameters:      {total_params:,}")
        print(f"Trainable Parameters:  {trainable_params:,}")
        print(f"Model Size (MB):       {total_params * 4 / 1024 / 1024:.2f}")
        print(f"Device:                {self.device}")
        print(f"Optimizer:             Adam (lr={self.optimizer.param_groups[0]['lr']})")
        print(f"Loss Function:         MSE")
        print("="*70 + "\n")

    def _save_checkpoint(self, epoch: int, val_loss: float, is_best: bool = False) -> None:
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
            'train_loss': self.history["train_loss"][-1],
            'history': self.history,
        }
        
        # Always save latest checkpoint
        latest_path = os.path.join(self.checkpoint_dir, 'latest_checkpoint.pt')
        torch.save(checkpoint, latest_path)
        
        # Save best checkpoint
        if is_best:
            best_path = os.path.join(self.checkpoint_dir, 'best_model.pt')
            torch.save(checkpoint, best_path)
            print(f"✓ New best model saved! (Val Loss: {val_loss:.6f})")

    def _update_live_plot(self) -> None:
        """Update and save the loss plot after each epoch."""
        plt.figure(figsize=(12, 6))
        
        epochs = range(1, len(self.history["train_loss"]) + 1)
        
        plt.plot(epochs, self.history["train_loss"], 'b-o', label='Training Loss', linewidth=2, markersize=6)
        plt.plot(epochs, self.history["val_loss"], 'r-s', label='Validation Loss', linewidth=2, markersize=6)
        
        # Mark the best validation loss
        best_epoch = self.history["val_loss"].index(min(self.history["val_loss"])) + 1
        best_val_loss = min(self.history["val_loss"])
        plt.plot(best_epoch, best_val_loss, 'g*', markersize=20, 
                label=f'Best (Epoch {best_epoch})', zorder=5)
        
        plt.title('Training and Validation Loss Over Epochs', fontsize=14, fontweight='bold')
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Loss (MSE)', fontsize=12)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        
        # Add text with current stats
        current_epoch = len(self.history["train_loss"])
        stats_text = f"Current Epoch: {current_epoch}\n"
        stats_text += f"Train Loss: {self.history['train_loss'][-1]:.6f}\n"
        stats_text += f"Val Loss: {self.history['val_loss'][-1]:.6f}\n"
        stats_text += f"Best Val Loss: {best_val_loss:.6f}"
        
        plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes,
                fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig("loss_plot.png", dpi=100)
        plt.close()

    def train(self, num_epochs: int) -> nn.Module:
        """Run the optimisation loop for ``num_epochs`` and return the trained model."""

        print("\n" + "="*70)
        print("STARTING TRAINING")
        print("="*70)
        print(f"Total Epochs: {num_epochs}")
        print(f"Training Batches per Epoch: {len(self.train_loader)}")
        print(f"Validation Batches per Epoch: {len(self.val_loader)}")
        print("="*70 + "\n")
        
        for epoch in range(num_epochs):
            # --- Training Phase ---
            self.model.train()
            running_train_loss = 0.0
            for batch_idx, (x_batch, y_batch) in enumerate(self.train_loader):
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                predictions = self.model(x_batch)
                loss = self.criterion(predictions, y_batch)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                running_train_loss += loss.item()

            epoch_train_loss = running_train_loss / max(1, len(self.train_loader))
            self.history["train_loss"].append(epoch_train_loss)

            # --- Validation Phase ---
            self.model.eval()
            running_val_loss = 0.0
            with torch.no_grad():
                for x_batch, y_batch in self.val_loader:
                    x_batch = x_batch.to(self.device)
                    y_batch = y_batch.to(self.device)
                    predictions = self.model(x_batch)
                    loss = self.criterion(predictions, y_batch)
                    running_val_loss += loss.item()

            epoch_val_loss = running_val_loss / max(1, len(self.val_loader))
            self.history["val_loss"].append(epoch_val_loss)

            # --- Checkpointing ---
            is_best = epoch_val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = epoch_val_loss
            
            self._save_checkpoint(epoch + 1, epoch_val_loss, is_best)
            
            # --- Live Plotting ---
            self._update_live_plot()

            # --- Print Progress ---
            status = "🌟 NEW BEST!" if is_best else ""
            print(
                f"Epoch [{epoch + 1:3d}/{num_epochs}] | "
                f"Train Loss: {epoch_train_loss:.6f} | "
                f"Val Loss: {epoch_val_loss:.6f} | "
                f"Best Val: {self.best_val_loss:.6f} {status}"
            )

        print("\n" + "="*70)
        print("TRAINING COMPLETE")
        print("="*70)
        print(f"Best Validation Loss: {self.best_val_loss:.6f}")
        print(f"Final Training Loss:  {self.history['train_loss'][-1]:.6f}")
        print(f"Final Validation Loss: {self.history['val_loss'][-1]:.6f}")
        print(f"Best model saved to: {os.path.join(self.checkpoint_dir, 'best_model.pt')}")
        print("="*70 + "\n")
        
        # Load best model before returning
        self._load_best_model()
        return self.model

    def _load_best_model(self) -> None:
        """Load the best model from checkpoint."""
        best_path = os.path.join(self.checkpoint_dir, 'best_model.pt')
        if os.path.exists(best_path):
            checkpoint = torch.load(best_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"✓ Loaded best model from epoch {checkpoint['epoch']}")

    def plot_loss(self) -> None:
        """Final loss plot (already done in real-time, but kept for compatibility)."""
        self._update_live_plot()
        print("Final loss plot saved to loss_plot.png")


# --- Evaluation ---------------------------------------------------------------------


class Evaluator:
    """Autoregressive evaluation utilities for the trained model."""

    def __init__(self, model: nn.Module, device: torch.device) -> None:
        self.model = model.to(device)
        self.device = device

    def _perform_prediction(
        self,
        prompt_df: pd.DataFrame,
        num_to_predict: int,
    ) -> pd.DataFrame:
        """Generate ``num_to_predict`` steps autoregressively from ``prompt_df``."""

        self.model.eval()
        model_input = torch.tensor(
            prompt_df[["Latitude", "Longitude", "SOG", "COG"]].values,
            dtype=torch.float32,
        ).unsqueeze(0).to(self.device)

        predicted_features = []
        with torch.no_grad():
            for _ in range(num_to_predict):
                prediction = self.model(model_input)
                next_input_vec_tensor = prediction[0, -1, :]
                predicted_features.append(next_input_vec_tensor.cpu().numpy())

                next_input_reshaped = next_input_vec_tensor.unsqueeze(0).unsqueeze(0)
                model_input = torch.cat([model_input[:, 1:, :], next_input_reshaped], dim=1)

        return pd.DataFrame(
            predicted_features,
            columns=["Pred_Lat", "Pred_Lon", "Pred_SOG", "Pred_COG"],
        )

    def _calculate_metrics(
        self,
        true_df: pd.DataFrame,
        pred_df: pd.DataFrame,
        t_prompt_end: pd.Timestamp,
    ) -> pd.DataFrame:
        """Return a small table with positional errors at key horizons."""

        results_df = pd.concat(
            [true_df.reset_index(drop=True), pred_df.reset_index(drop=True)], axis=1
        )

        def _safe_idx(mask: pd.Series) -> int:
            try:
                return mask.idxmax()
            except ValueError:
                return -1

        t_20m_idx = _safe_idx(true_df["Timestamp"] >= (t_prompt_end + pd.Timedelta(minutes=20)))
        t_1h_idx = _safe_idx(true_df["Timestamp"] >= (t_prompt_end + pd.Timedelta(hours=1)))
        t_last_idx = len(true_df) - 1

        indices_to_check = {
            "20 Minutes": t_20m_idx,
            "1 Hour": t_1h_idx,
            "Last Point": t_last_idx,
        }

        error_table = []
        for name, idx in indices_to_check.items():
            if 0 <= idx < len(results_df):
                row = results_df.iloc[idx]
                error = haversine_distance(
                    row["Latitude"],
                    row["Longitude"],
                    row["Pred_Lat"],
                    row["Pred_Lon"],
                )
                error_table.append({"Time Horizon": name, "Error (meters)": f"{error:.2f}"})

        return pd.DataFrame(error_table)

    def _plot_map(
        self,
        full_true_df: pd.DataFrame,
        pred_df: pd.DataFrame,
        start_pred_point: Tuple[float, float],
    ) -> None:
        """Create a Folium map comparing true and predicted trajectories."""

        print("\nGenerating Folium map...")
        m = folium.Map()

        true_points = list(zip(full_true_df["Latitude"], full_true_df["Longitude"]))
        pred_points = list(zip(pred_df["Pred_Lat"], pred_df["Pred_Lon"]))

        folium.PolyLine(true_points, color="blue", weight=4, opacity=0.8, popup="True Path").add_to(m)
        folium.PolyLine(pred_points, color="red", weight=4, opacity=0.8, popup="Predicted Path").add_to(m)
        folium.Marker(
            location=start_pred_point,
            popup="Start of Prediction",
            icon=folium.Icon(color="green"),
        ).add_to(m)

        m.fit_bounds(m.get_bounds())
        map_path = "prediction_map.html"
        m.save(map_path)
        print(f"Map saved to {map_path}")

    def evaluate_and_plot(
        self,
        test_files: Sequence[str],
        history_len: int,
        file_to_plot: int = 0,
    ) -> None:
        """Run autoregressive evaluation on one test trajectory and plot the results."""

        if not test_files:
            print("No test files available for evaluation.")
            return

        if file_to_plot >= len(test_files):
            raise IndexError(
                f"Requested file index {file_to_plot} but only {len(test_files)} test files are available."
            )

        test_file = test_files[file_to_plot]
        print(f"\n--- Evaluating on Test File: {test_file} ---")

        df = pd.read_parquet(
            test_file,
            columns=["Timestamp", "Latitude", "Longitude", "SOG", "COG"],
        )
        df["Timestamp"] = pd.to_datetime(df["Timestamp"])

        t_start = df["Timestamp"].iloc[0]
        t_prompt_end = t_start + pd.Timedelta(hours=1)

        prompt_df = df[df["Timestamp"] <= t_prompt_end]
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

        pred_df = self._perform_prediction(prompt_df, num_to_predict)
        error_table_df = self._calculate_metrics(true_df, pred_df, t_prompt_end)

        print("\n--- Prediction Error Table ---")
        if error_table_df.empty:
            print("Not enough data to compute error metrics.")
        else:
            print(error_table_df.to_string(index=False))

        start_pred_point_coords = (true_df.iloc[0]["Latitude"], true_df.iloc[0]["Longitude"])
        self._plot_map(df, pred_df, start_pred_point_coords)