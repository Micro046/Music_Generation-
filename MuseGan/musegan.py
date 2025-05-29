"""
Main MuseGAN class for training and generation
"""

import torch
import torch.optim as optim
import os
from config import *
from models import Generator, Critic


class MuseGAN:
    """Main MuseGAN training and generation class"""

    def __init__(self, generator=None, critic=None, latent_dim=Z_DIM,
                 critic_steps=CRITIC_STEPS, gp_weight=GP_WEIGHT, device=DEVICE):

        # Initialize models
        if generator is None:
            generator = Generator()
        if critic is None:
            critic = Critic()

        self.generator = generator.to(device)
        self.critic = critic.to(device)

        # Training parameters
        self.latent_dim = latent_dim
        self.critic_steps = critic_steps
        self.gp_weight = gp_weight
        self.device = device

        # Initialize optimizers
        self.g_optimizer = optim.Adam(
            self.generator.parameters(),
            lr=GENERATOR_LEARNING_RATE,
            betas=(ADAM_BETA_1, ADAM_BETA_2)
        )
        self.c_optimizer = optim.Adam(
            self.critic.parameters(),
            lr=CRITIC_LEARNING_RATE,
            betas=(ADAM_BETA_1, ADAM_BETA_2)
        )

        # Training metrics
        self.training_history = {
            'c_loss': [],
            'c_wass_loss': [],
            'c_gp': [],
            'g_loss': []
        }

    def gradient_penalty(self, real_samples, fake_samples):
        """Calculate gradient penalty for WGAN-GP"""
        batch_size = real_samples.size(0)
        alpha = torch.rand(batch_size, 1, 1, 1, 1).to(self.device)

        # Interpolated samples
        interpolated = alpha * real_samples + (1 - alpha) * fake_samples
        interpolated.requires_grad_(True)

        # Critic score for interpolated samples
        critic_interpolated = self.critic(interpolated)

        # Calculate gradients
        gradients = torch.autograd.grad(
            outputs=critic_interpolated,
            inputs=interpolated,
            grad_outputs=torch.ones_like(critic_interpolated),
            create_graph=True,
            retain_graph=True
        )[0]

        # Calculate gradient penalty
        gradients = gradients.view(batch_size, -1)
        gradient_norm = gradients.norm(2, dim=1)
        gradient_penalty = torch.mean((gradient_norm - 1) ** 2)

        return gradient_penalty

    def generate_random_inputs(self, batch_size):
        """Generate random latent vectors for all input types"""
        chords_z = torch.randn(batch_size, self.latent_dim).to(self.device)
        style_z = torch.randn(batch_size, self.latent_dim).to(self.device)
        melody_z = torch.randn(batch_size, N_TRACKS, self.latent_dim).to(self.device)
        groove_z = torch.randn(batch_size, N_TRACKS, self.latent_dim).to(self.device)

        return [chords_z, style_z, melody_z, groove_z]

    def train_step(self, real_data):
        """Single training step for both critic and generator"""
        batch_size = real_data.size(0)

        # Train Critic multiple times
        c_loss_total = 0
        c_wass_loss_total = 0
        c_gp_total = 0

        for _ in range(self.critic_steps):
            self.c_optimizer.zero_grad()

            # Generate fake data
            random_inputs = self.generate_random_inputs(batch_size)
            fake_data = self.generator(random_inputs)

            # Critic scores
            real_score = self.critic(real_data)
            fake_score = self.critic(fake_data.detach())

            # Wasserstein loss
            c_wass_loss = torch.mean(fake_score) - torch.mean(real_score)

            # Gradient penalty
            gp = self.gradient_penalty(real_data, fake_data.detach())

            # Total critic loss
            c_loss = c_wass_loss + self.gp_weight * gp

            c_loss.backward()
            self.c_optimizer.step()

            # Accumulate losses
            c_loss_total += c_loss.item()
            c_wass_loss_total += c_wass_loss.item()
            c_gp_total += gp.item()

        # Average critic losses
        c_loss_avg = c_loss_total / self.critic_steps
        c_wass_loss_avg = c_wass_loss_total / self.critic_steps
        c_gp_avg = c_gp_total / self.critic_steps

        # Train Generator
        self.g_optimizer.zero_grad()

        random_inputs = self.generate_random_inputs(batch_size)
        fake_data = self.generator(random_inputs)
        fake_score = self.critic(fake_data)

        g_loss = -torch.mean(fake_score)

        g_loss.backward()
        self.g_optimizer.step()

        # Return loss dictionary
        losses = {
            'c_loss': c_loss_avg,
            'c_wass_loss': c_wass_loss_avg,
            'c_gp': c_gp_avg,
            'g_loss': g_loss.item()
        }

        # Update training history
        for key, value in losses.items():
            self.training_history[key].append(value)

        return losses

    def generate_music(self, num_samples=1):
        """Generate music samples"""
        self.generator.eval()
        with torch.no_grad():
            random_inputs = self.generate_random_inputs(num_samples)
            generated_music = self.generator(random_inputs)
        self.generator.train()
        return generated_music.cpu().numpy()

    def save_model(self, path):
        """Save model weights and optimizer states"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'generator_state_dict': self.generator.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'g_optimizer_state_dict': self.g_optimizer.state_dict(),
            'c_optimizer_state_dict': self.c_optimizer.state_dict(),
            'training_history': self.training_history
        }, path)
        print(f"Model saved to {path}")

    def load_model(self, path):
        """Load model weights and optimizer states"""
        if os.path.exists(path):
            checkpoint = torch.load(path, map_location=self.device)
            self.generator.load_state_dict(checkpoint['generator_state_dict'])
            self.critic.load_state_dict(checkpoint['critic_state_dict'])
            self.g_optimizer.load_state_dict(checkpoint['g_optimizer_state_dict'])
            self.c_optimizer.load_state_dict(checkpoint['c_optimizer_state_dict'])

            if 'training_history' in checkpoint:
                self.training_history = checkpoint['training_history']

            print(f"Model loaded from {path}")
        else:
            print(f"No model found at {path}")

    def get_training_history(self):
        """Return training history for plotting"""
        return self.training_history

    def set_device(self, device):
        """Move models to specified device"""
        self.device = device
        self.generator = self.generator.to(device)
        self.critic = self.critic.to(device)