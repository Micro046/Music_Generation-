"""
Data processing utilities for MuseGAN
"""

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from config import *


def load_bach_chorales(file_path=DATA_PATH):
    """
    Load and preprocess Bach chorales data

    Args:
        file_path (str): Path to the .npz data file

    Returns:
        numpy.ndarray: Processed data ready for training
    """
    try:
        with np.load(file_path, encoding="bytes", allow_pickle=True) as f:
            data = f["train"]

        print(f"{len(data)} chorales loaded from dataset")

        # Display sample info
        if len(data) > 0:
            chorale = data[0]
            n_beats, n_tracks = chorale.shape
            print(f"Sample chorale shape: {chorale.shape} (beats: {n_beats}, tracks: {n_tracks})")
            print("\nFirst 8 steps of chorale 0:")
            print(chorale[:8])

        return data

    except FileNotFoundError:
        print(f"Data file not found at {file_path}")
        print("Please make sure the Bach chorales dataset is available.")
        return None
    except Exception as e:
        print(f"Error loading data: {e}")
        return None


def preprocess_data(data):
    """
    Preprocess the Bach chorales data for training

    Args:
        data: Raw data from load_bach_chorales()

    Returns:
        numpy.ndarray: Preprocessed binary data
    """
    if data is None:
        return None

    # Extract two bars from each chorale
    two_bars = np.array([x[:(N_STEPS_PER_BAR * N_BARS)] for x in data])

    # Handle NaN values
    two_bars = np.array(np.nan_to_num(two_bars, nan=MAX_PITCH), dtype=int)

    # Reshape to [n_songs, n_bars, n_steps_per_bar, n_tracks]
    n_songs = len(data)
    two_bars = two_bars.reshape([n_songs, N_BARS, N_STEPS_PER_BAR, N_TRACKS])

    print(f"Two bars shape: {two_bars.shape}")

    # Convert to one-hot encoding
    data_binary = np.eye(N_PITCHES)[two_bars]

    # Convert 0s to -1s (for better GAN training)
    data_binary[data_binary == 0] = -1

    # Transpose to [batch, bars, steps, pitches, tracks]
    data_binary = data_binary.transpose([0, 1, 2, 4, 3])

    print(f"Data binary shape: {data_binary.shape}")
    print(f"Data range: [{data_binary.min()}, {data_binary.max()}]")

    return data_binary


def create_dataloader(data_binary, batch_size=BATCH_SIZE, shuffle=True):
    """
    Create PyTorch DataLoader from preprocessed data

    Args:
        data_binary: Preprocessed binary data
        batch_size (int): Batch size for training
        shuffle (bool): Whether to shuffle the data

    Returns:
        torch.utils.data.DataLoader: DataLoader for training
    """
    if data_binary is None:
        return None

    # Convert to PyTorch tensor
    data_tensor = torch.FloatTensor(data_binary)
    print(f"Created tensor with shape: {data_tensor.shape}")

    # Create dataset and dataloader
    dataset = TensorDataset(data_tensor)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=True,  # Drop last incomplete batch
        num_workers=0  # Set to 0 for Windows compatibility
    )

    print(f"DataLoader created with {len(dataloader)} batches")
    return dataloader


def prepare_training_data(file_path=DATA_PATH, batch_size=BATCH_SIZE):
    """
    Complete data preparation pipeline

    Args:
        file_path (str): Path to the data file
        batch_size (int): Batch size for training

    Returns:
        torch.utils.data.DataLoader: Ready-to-use DataLoader
    """
    print("Loading Bach chorales data...")
    raw_data = load_bach_chorales(file_path)

    if raw_data is None:
        return None

    print("Preprocessing data...")
    processed_data = preprocess_data(raw_data)

    if processed_data is None:
        return None

    print("Creating DataLoader...")
    dataloader = create_dataloader(processed_data, batch_size)

    print("Data preparation complete!")
    return dataloader


def get_data_info():
    """
    Get information about the data configuration

    Returns:
        dict: Data configuration information
    """
    return {
        'n_bars': N_BARS,
        'n_steps_per_bar': N_STEPS_PER_BAR,
        'n_pitches': N_PITCHES,
        'n_tracks': N_TRACKS,
        'max_pitch': MAX_PITCH,
        'batch_size': BATCH_SIZE
    }


def validate_data_shape(data):
    """
    Validate that data has the expected shape

    Args:
        data: Data to validate

    Returns:
        bool: True if data shape is valid
    """
    if data is None:
        return False

    expected_shape = (None, N_BARS, N_STEPS_PER_BAR, N_PITCHES, N_TRACKS)
    actual_shape = data.shape

    if len(actual_shape) != 5:
        print(f"Expected 5D data, got {len(actual_shape)}D")
        return False

    if actual_shape[1:] != expected_shape[1:]:
        print(f"Expected shape {expected_shape}, got {actual_shape}")
        return False

    return True

if __name__ == "__main__":
    # Test data loading and preprocessing
    print("Testing data preparation...")

    dataloader = prepare_training_data()

    if dataloader is not None:
        print("\nTesting DataLoader...")
        for batch_idx, (batch_data,) in enumerate(dataloader):
            print(f"Batch {batch_idx}: {batch_data.shape}")
            if batch_idx >= 2:  # Only show first 3 batches
                break

        print(f"\nTotal batches: {len(dataloader)}")
        print("Data preparation test completed successfully!")
    else:
        print("Data preparation failed!")
