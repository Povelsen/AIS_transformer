"""Data loading, training, and evaluation helpers for the AIS Transformer."""

from __future__ import annotations

import os
from typing import List, Sequence, Tuple, Optional

import folium
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


# --- Helper utilities -------------------------------------------------------------------------

def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Return the distance between two lat/lon points in meters."""
    return haversine((lat1, lon1), (lat2, lon2), unit=Unit.METERS)


def circular_loss(pred_cog: torch.Tensor, true_cog: torch.Tensor) -> torch.Tensor:
    """Compute loss for circular angle (COG) in degrees.
    
    Handles wraparound: difference between 1° and 359° is 2°, not 358°.
    """
    diff = (pred_cog - true_cog + 180) % 360 - 180
    return (diff ** 2).mean()


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

        # Share normalization statistics
        val_dataset.mean = train_dataset.mean
        val_dataset.std = train_dataset.std

        pin_memory = torch.cuda.is_available()
        persistent_workers = num_workers > 0
        loader_extras = {}
        if num_workers > 0:
            loader_extras["prefetch_factor"] = 2

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            **loader_extras,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            **loader_extras,
        )
        return train_loader, val_loader, self.test_files


class VesselTrajectoryDataset(Dataset):
    """Windowed trajectory dataset with normalization."""

    def __init__(self, segment_files: Sequence[str], history_len: int, future_len: int, stride: int = 5) -> None:
        self.segment_files = list(segment_files)
        self.history_len = history_len
        self.future_len = future_len
        self.window_len = history_len + future_len
        self.stride = stride

        self.features = ["Latitude", "Longitude", "SOG", "COG"]
        self.targets = ["Latitude", "Longitude", "SOG", "COG"]

        self.samples: List[Tuple[str, int]] = []
        self._cache: dict[str, torch.Tensor] = {}
        
        # Normalization statistics (will be computed)
        self.mean: Optional[torch.Tensor] = None
        self.std: Optional[torch.Tensor] = None
        
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
        
        # Compute normalization statistics
        self._compute_normalization_stats()

    def _compute_normalization_stats(self) -> None:
        """Compute mean and std for feature normalization."""
        print("Computing normalization statistics...")
        
        # Sample data from first 100 segments (or all if fewer)
        sample_files = self.segment_files[:min(100, len(self.segment_files))]
        all_data = []
        
        for file_path in sample_files:
            try:
                table = pq.read_table(file_path, columns=self.features, memory_map=True)
                data = np.column_stack([
                    table.column(name).combine_chunks().to_numpy(zero_copy_only=False).astype(np.float32)
                    for name in self.features
                ])
                all_data.append(data)
            except Exception as exc:
                print(f"Warning: Could not read {file_path} for normalization: {exc}")
                continue
        
        if not all_data:
            raise RuntimeError("Could not load any data for normalization statistics")
        
        all_data = np.vstack(all_data)
        self.mean = torch.tensor(all_data.mean(axis=0), dtype=torch.float32)
        self.std = torch.tensor(all_data.std(axis=0), dtype=torch.float32)
        
        # Prevent division by zero
        self.std = torch.clamp(self.std, min=1e-6)
        
        print(f"Normalization stats computed:")
        print(f"  Mean: {self.mean.numpy()}")
        print(f"  Std:  {self.std.numpy()}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        file_path, start_idx = self.samples[idx]

        # Lazy-load and cache the parquet for this file_path
        if file_path not in self._cache:
            table = pq.read_table(file_path, columns=self.features, memory_map=True)
            stacked = np.column_stack([
                table.column(name).combine_chunks().to_numpy(zero_copy_only=False).astype(np.float32, copy=False)
                for name in self.features
            ])
            self._cache[file_path] = torch.from_numpy(stacked).contiguous()

        cached_tensor = self._cache[file_path]
        window = cached_tensor[start_idx : start_idx + self.window_len]

        # Split into history and future
        history = window[: self.history_len]             # [H, 4]
        future  = window[self.history_len : ]            # [F, 4]  (F = future_len)

        # Normalize
        history = (history - self.mean) / self.std
        future  = (future  - self.mean) / self.std

        # Return normalized history and future
        return history, future


# --- Training ------------------------------------------------------------------------


class Trainer:
    """Handles the model training loop with checkpointing, live plotting, and improved loss."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        learning_rate: float,
        device: torch.device,
        checkpoint_dir: str = "checkpoints",
        normalization_stats: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> None:
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        # Store normalization stats for denormalization during evaluation
        self.norm_mean = normalization_stats[0] if normalization_stats else None
        self.norm_std = normalization_stats[1] if normalization_stats else None
        
        # Mixed loss: MSE for position/speed, circular loss for COG
        self.mse_criterion = nn.MSELoss()
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=2, # verbose=True
        )
        
        self.history = {"train_loss": [], "val_loss": []}
        self.checkpoint_dir = checkpoint_dir
        self.best_val_loss = float('inf')
        
        # Create checkpoint directory
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # Print model statistics
        self._print_model_stats()

    def _compute_loss(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute combined loss: MSE for Lat/Lon/SOG, circular loss for COG."""
        
        # MSE loss for Lat, Lon, SOG (indices 0, 1, 2)
        mse_loss = self.mse_criterion(predictions[:, :, :3], targets[:, :, :3])
        
        # Circular loss for COG (index 3) - denormalize first
        pred_cog = predictions[:, :, 3]
        true_cog = targets[:, :, 3]
        
        # Denormalize COG if normalization is used
        if self.norm_mean is not None:
            mean_cog = self.norm_mean[3].to(self.device)
            std_cog = self.norm_std[3].to(self.device)
            pred_cog = pred_cog * std_cog + mean_cog
            true_cog = true_cog * std_cog + mean_cog
        
        cog_loss = circular_loss(pred_cog, true_cog)
        
        # Weighted combination (COG is less important than position)
        total_loss = mse_loss + 0.1 * cog_loss
        
        return total_loss

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
        print(f"Loss Function:         MSE + Circular Loss for COG")
        print(f"LR Scheduler:          ReduceLROnPlateau")
        print("="*70 + "\n")

    def _save_checkpoint(self, epoch: int, val_loss: float, is_best: bool = False) -> None:
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_loss': val_loss,
            'train_loss': self.history["train_loss"][-1],
            'history': self.history,
            'norm_mean': self.norm_mean,
            'norm_std': self.norm_std,
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
        plt.ylabel('Loss (Combined MSE + Circular)', fontsize=12)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        
        # Add text with current stats
        current_epoch = len(self.history["train_loss"])
        current_lr = self.optimizer.param_groups[0]['lr']
        stats_text = f"Current Epoch: {current_epoch}\n"
        stats_text += f"Learning Rate: {current_lr:.6f}\n"
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
            self.model.train()
            running_train_loss = 0.0

            train_iter = tqdm(
                self.train_loader,
                desc=f"Epoch {epoch + 1}/{num_epochs} [TRAIN]",
                leave=False
            )

            for x_batch, y_future in train_iter:
                # x_batch: history [B, H, 4]
                # y_future: future [B, F, 4]
                x_batch = x_batch.to(self.device, non_blocking=True)
                y_future = y_future.to(self.device, non_blocking=True)

                # Build decoder input by shifting the future:
                # first token = last history point, then all but last future point
                # decoder_input shape: [B, F, 4]
                start_token = x_batch[:, -1:, :]              # [B, 1, 4]
                if y_future.size(1) > 1:
                    dec_rest = y_future[:, :-1, :]            # [B, F-1, 4]
                    decoder_input = torch.cat([start_token, dec_rest], dim=1)
                else:
                    decoder_input = start_token               # F=1 case

                # Forward: encoder–decoder
                predictions = self.model(x_batch, decoder_input)   # [B, F, 4]

                # Loss against the true future
                loss = self._compute_loss(predictions, y_future)

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()   

                running_train_loss += loss.item()
                train_iter.set_postfix(loss=loss.item())


            epoch_train_loss = running_train_loss / max(1, len(self.train_loader))
            self.history["train_loss"].append(epoch_train_loss)

            # --- Validation Phase ---
            self.model.eval()
            running_val_loss = 0.0
            with torch.no_grad():
                val_iter = tqdm(
                    self.val_loader,
                    desc=f"Epoch {epoch + 1}/{num_epochs} [VAL]",
                    leave=False
                )
                for x_batch, y_future in val_iter:
                    x_batch = x_batch.to(self.device, non_blocking=True)
                    y_future = y_future.to(self.device, non_blocking=True)

                    start_token = x_batch[:, -1:, :]
                    if y_future.size(1) > 1:
                        dec_rest = y_future[:, :-1, :]
                        decoder_input = torch.cat([start_token, dec_rest], dim=1)
                    else:
                        decoder_input = start_token

                    predictions = self.model(x_batch, decoder_input)
                    loss = self._compute_loss(predictions, y_future)
                    running_val_loss += loss.item()
                    val_iter.set_postfix(loss=loss.item())

            epoch_val_loss = running_val_loss / max(1, len(self.val_loader))
            self.history["val_loss"].append(epoch_val_loss)

            # --- Learning Rate Scheduling ---
            old_lr = self.optimizer.param_groups[0]['lr']
            self.scheduler.step(epoch_val_loss)
            new_lr = self.optimizer.param_groups[0]['lr']
            if new_lr != old_lr:
                print(f"LR reduced: {old_lr:.6f} → {new_lr:.6f}")


            # --- Checkpointing ---
            is_best = epoch_val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = epoch_val_loss
            
            self._save_checkpoint(epoch + 1, epoch_val_loss, is_best)
            
            # --- Live Plotting ---
            self._update_live_plot()

            # --- Print Progress ---
            status = "🌟 NEW BEST!" if is_best else ""
            current_lr = self.optimizer.param_groups[0]['lr']
            print(
                f"Epoch [{epoch + 1:3d}/{num_epochs}] | "
                f"Train Loss: {epoch_train_loss:.6f} | "
                f"Val Loss: {epoch_val_loss:.6f} | "
                f"Best Val: {self.best_val_loss:.6f} | "
                f"LR: {current_lr:.6f} {status}"
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
    """Autoregressive evaluation utilities for the trained model with enhanced metrics."""

    def __init__(
        self, 
        model: nn.Module, 
        device: torch.device,
        normalization_stats: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> None:
        self.model = model.to(device)
        self.device = device
        
        # Store normalization stats for denormalization
        self.norm_mean = normalization_stats[0] if normalization_stats else None
        self.norm_std = normalization_stats[1] if normalization_stats else None

    def _denormalize(self, data: np.ndarray) -> np.ndarray:
        """Denormalize predictions back to original scale."""
        if self.norm_mean is not None:
            mean = self.norm_mean.cpu().numpy()
            std = self.norm_std.cpu().numpy()
            return data * std + mean
        return data
        
    def _perform_prediction(
        self,
        prompt_df: pd.DataFrame,
        num_to_predict: int,
    ) -> pd.DataFrame:
        """Generate ``num_to_predict`` steps autoregressively from ``prompt_df``.

        Uses encoder–decoder properly:
        - Encoder sees full history
        - Decoder starts from last history point and grows one step at a time
        """

        self.model.eval()

        # 1) Prepare normalized history
        input_data = prompt_df[["Latitude", "Longitude", "SOG", "COG"]].values.astype(np.float32)
        if self.norm_mean is not None:
            mean = self.norm_mean.cpu().numpy()
            std = self.norm_std.cpu().numpy()
            input_data = (input_data - mean) / std

        src_seq = torch.tensor(input_data, dtype=torch.float32).unsqueeze(0).to(
            self.device, non_blocking=True
        )  # [1, H, 4]

        # 2) Initial decoder input: last history point
        decoder_input = src_seq[:, -1:, :]  # [1, 1, 4]

        predicted_features = []

        # 3) Autoregressive loop
        with torch.no_grad():
            for _ in range(num_to_predict):
                # Forward pass: encoder–decoder
                preds = self.model(src_seq, decoder_input)      # [1, L, 4]
                next_step = preds[:, -1, :]                     # [1, 4]

                predicted_features.append(next_step.squeeze(0).cpu().numpy())

                # Append predicted step to decoder input
                decoder_input = torch.cat(
                    [decoder_input, next_step.unsqueeze(1)], dim=1
                )  # grow from [1, t, 4] → [1, t+1, 4]

        # 4) Denormalize predictions
        predictions_array = np.array(predicted_features)   # [T, 4]
        predictions_denorm = self._denormalize(predictions_array)

        return pd.DataFrame(
            predictions_denorm,
            columns=["Pred_Lat", "Pred_Lon", "Pred_SOG", "Pred_COG"],
        )


        # Denormalize predictions
        predictions_array = np.array(predicted_features)
        predictions_denorm = self._denormalize(predictions_array)
        
        return pd.DataFrame(
            predictions_denorm,
            columns=["Pred_Lat", "Pred_Lon", "Pred_SOG", "Pred_COG"],
        )

    def _calculate_metrics(
        self,
        true_df: pd.DataFrame,
        pred_df: pd.DataFrame,
        t_prompt_end: pd.Timestamp,
    ) -> pd.DataFrame:
        """Return a comprehensive table with multiple evaluation metrics."""

        results_df = pd.concat(
            [true_df.reset_index(drop=True), pred_df.reset_index(drop=True)], axis=1
        )

        # Calculate distance errors for all points
        all_errors = []
        for i in range(len(results_df)):
            row = results_df.iloc[i]
            error = haversine_distance(
                row["Latitude"],
                row["Longitude"],
                row["Pred_Lat"],
                row["Pred_Lon"],
            )
            all_errors.append(error)
        
        # Calculate speed and heading errors
        sog_errors = np.abs(results_df["SOG"].values - results_df["Pred_SOG"].values)
        
        # Circular difference for COG
        cog_true = results_df["COG"].values
        cog_pred = results_df["Pred_COG"].values
        cog_errors = np.abs((cog_pred - cog_true + 180) % 360 - 180)

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
                error_table.append({
                    "Time Horizon": name,
                    "Position Error (m)": f"{all_errors[idx]:.2f}",
                    "Speed Error (m/s)": f"{sog_errors[idx]:.3f}",
                    "Heading Error (°)": f"{cog_errors[idx]:.2f}",
                })
        
        # Add aggregate metrics
        ade = np.mean(all_errors)  # Average Displacement Error
        fde = all_errors[-1]  # Final Displacement Error
        
        error_table.append({
            "Time Horizon": "Average (ADE)",
            "Position Error (m)": f"{ade:.2f}",
            "Speed Error (m/s)": f"{np.mean(sog_errors):.3f}",
            "Heading Error (°)": f"{np.mean(cog_errors):.2f}",
        })
        
        error_table.append({
            "Time Horizon": "Final (FDE)",
            "Position Error (m)": f"{fde:.2f}",
            "Speed Error (m/s)": f"{sog_errors[-1]:.3f}",
            "Heading Error (°)": f"{cog_errors[-1]:.2f}",
        })

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

        print("\n" + "="*70)
        print("PREDICTION METRICS")
        print("="*70)
        if error_table_df.empty:
            print("Not enough data to compute error metrics.")
        else:
            print(error_table_df.to_string(index=False))
        print("="*70 + "\n")

        start_pred_point_coords = (true_df.iloc[0]["Latitude"], true_df.iloc[0]["Longitude"])
        self._plot_map(df, pred_df, start_pred_point_coords)