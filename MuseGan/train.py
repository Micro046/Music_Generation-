"""
Training script for MuseGAN
"""

import os
import time
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
from config import *
from models import Generator, Critic
from musegan import MuseGAN
from data_utils import prepare_training_data
from musegan_utils import notes_to_midi


def create_directories():
    """Create necessary directories for training"""
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)


def plot_training_history(history, save_path=None):
    """Plot training history"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('MuseGAN Training History', fontsize=16)

    # Critic Loss
    axes[0, 0].plot(history['c_loss'], label='Critic Loss')
    axes[0, 0].set_title('Critic Loss')
    axes[0, 0].set_xlabel('Batch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # Generator Loss
    axes[0, 1].plot(history['g_loss'], label='Generator Loss', color='orange')
    axes[0, 1].set_title('Generator Loss')
    axes[0, 1].set_xlabel('Batch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    # Wasserstein Loss
    axes[1, 0].plot(history['c_wass_loss'], label='Wasserstein Loss', color='green')
    axes[1, 0].set_title('Wasserstein Loss')
    axes[1, 0].set_xlabel('Batch')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].legend()
    axes[1, 0].grid(True)

    # Gradient Penalty
    axes[1, 1].plot(history['c_gp'], label='Gradient Penalty', color='red')
    axes[1, 1].set_title('Gradient Penalty')
    axes[1, 1].set_xlabel('Batch')
    axes[1, 1].set_ylabel('Loss')
    axes[1, 1].legend()
    axes[1, 1].grid(True)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    if save_path:
        plt.savefig(save_path)
    plt.show()


if __name__ == '__main__':
    create_directories()

    print("Preparing data...")
    dataloader = prepare_training_data()
    if dataloader is None:
        print("Data loading failed!")
        exit(1)

    print("Initializing MuseGAN model...")
    musegan = MuseGAN()
    print("Starting training...")

    total_epochs = EPOCHS
    save_interval = 100  # epochs
    batches_per_epoch = len(dataloader)

    for epoch in range(total_epochs):
        batch_pbar = tqdm(enumerate(dataloader), total=batches_per_epoch, desc=f"Epoch {epoch + 1}/{total_epochs}")
        for batch_idx, (real_batch,) in batch_pbar:
            real_batch = real_batch.to(DEVICE)
            losses = musegan.train_step(real_batch)
            batch_pbar.set_postfix(losses)
        if (epoch + 1) % save_interval == 0:
            musegan.save_model(os.path.join(CHECKPOINT_DIR, f'checkpoint_epoch_{epoch + 1}.pth'))

    # Save final model
    musegan.save_model(os.path.join(CHECKPOINT_DIR, 'final_checkpoint.pth'))

    # Plot training history
    history = musegan.get_training_history()
    plot_training_history(history, save_path=os.path.join(LOGS_DIR, 'training_history.png'))
