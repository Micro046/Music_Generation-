import torch
import torch.nn as nn
from config import *

# Helper function for conv3d
def conv3d_block(in_channels, out_channels, kernel_size, stride=1, padding=0):
    return nn.Sequential(
        nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding),
        nn.LeakyReLU(0.2, inplace=True)
    )

def conv2d_transpose_block(in_channels, out_channels, kernel_size, stride=1, padding=0,
                           activation='relu', batch_norm=True):
    layers = [nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding)]
    if batch_norm:
        layers.append(nn.BatchNorm2d(out_channels, momentum=0.1))
    if activation == 'relu':
        layers.append(nn.ReLU(inplace=True))
    elif activation == 'tanh':
        layers.append(nn.Tanh())
    return nn.Sequential(*layers)

class TemporalNetwork(nn.Module):
    def __init__(self, z_dim=Z_DIM, n_bars=N_BARS):
        super().__init__()
        self.z_dim = z_dim
        self.n_bars = n_bars
        self.conv1 = conv2d_transpose_block(z_dim, 1024, (2, 1), stride=(1, 1), padding=(0, 0))
        self.conv2 = conv2d_transpose_block(1024, z_dim, (n_bars - 1, 1), stride=(1, 1), padding=(0, 0))
    def forward(self, x):
        x = x.view(x.size(0), self.z_dim, 1, 1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), self.n_bars, self.z_dim)
        return x

class BarGenerator(nn.Module):
    def __init__(self, z_dim=Z_DIM, n_steps_per_bar=N_STEPS_PER_BAR, n_pitches=N_PITCHES):
        super().__init__()
        self.n_steps_per_bar = n_steps_per_bar
        self.n_pitches = n_pitches

        self.dense = nn.Sequential(
            nn.Linear(z_dim * 4, 1024 * 2 * 1),  # <---- Output 2048 for [1024, 2, 1]
            nn.BatchNorm1d(1024 * 2 * 1, momentum=0.1),
            nn.ReLU(inplace=True)
        )

        self.deconv1 = conv2d_transpose_block(1024, 512, (2, 1), stride=(2, 1), padding=(0, 0))
        self.deconv2 = conv2d_transpose_block(512, 256, (2, 1), stride=(2, 1), padding=(0, 0))
        self.deconv3 = conv2d_transpose_block(256, 128, (2, 1), stride=(2, 1), padding=(0, 0))
        self.deconv4 = conv2d_transpose_block(128, 128, (1, 7), stride=(1, 7), padding=(0, 0))
        self.deconv5 = conv2d_transpose_block(128, 1, (1, 12), stride=(1, 12), padding=(0, 0), activation='tanh', batch_norm=False)

    def forward(self, x):
        x = self.dense(x)                  # [B, 2048]
        x = x.view(x.size(0), 1024, 2, 1)  # [B, 1024, 2, 1]
        x = self.deconv1(x)                # [B, 512, 4, 1]
        x = self.deconv2(x)                # [B, 256, 8, 1]
        x = self.deconv3(x)                # [B, 128, 16, 1]
        x = self.deconv4(x)                # [B, 128, 16, 7]
        x = self.deconv5(x)                # [B, 1, 16, 84]
        x = x.unsqueeze(-1)                # [B, 1, 16, 84, 1]
        return x


class Generator(nn.Module):
    def __init__(self, z_dim=Z_DIM, n_tracks=N_TRACKS, n_bars=N_BARS):
        super().__init__()
        self.z_dim = z_dim
        self.n_tracks = n_tracks
        self.n_bars = n_bars
        self.chords_temp_network = TemporalNetwork(z_dim, n_bars)
        self.melody_temp_networks = nn.ModuleList([TemporalNetwork(z_dim, n_bars) for _ in range(n_tracks)])
        self.bar_generators = nn.ModuleList([BarGenerator(z_dim) for _ in range(n_tracks)])

    def forward(self, inputs):
        chords_input, style_input, melody_input, groove_input = inputs
        batch_size = chords_input.size(0)
        chords_over_time = self.chords_temp_network(chords_input)
        melody_over_time = [self.melody_temp_networks[track](melody_input[:, track, :])
                            for track in range(self.n_tracks)]
        bars_output = []
        for bar in range(self.n_bars):
            track_outputs = []
            c = chords_over_time[:, bar, :]
            s = style_input
            for track in range(self.n_tracks):
                m = melody_over_time[track][:, bar, :]
                g = groove_input[:, track, :]
                z_input = torch.cat([c, s, m, g], dim=1)
                track_output = self.bar_generators[track](z_input)
                track_outputs.append(track_output)
            bar_output = torch.cat(track_outputs, dim=-1)
            bars_output.append(bar_output)
        generator_output = torch.cat(bars_output, dim=1)
        return generator_output

class Critic(nn.Module):
    def __init__(self, n_bars=N_BARS, n_steps_per_bar=N_STEPS_PER_BAR, n_pitches=N_PITCHES, n_tracks=N_TRACKS):
        super().__init__()
        self.conv_layers = nn.Sequential(
            conv3d_block(n_tracks, 128, (2, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0)),
            conv3d_block(128, 128, (n_bars - 1, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0)),
            conv3d_block(128, 128, (1, 1, 12), stride=(1, 1, 12), padding=(0, 0, 0)),
            conv3d_block(128, 128, (1, 1, 7), stride=(1, 1, 7), padding=(0, 0, 0)),
            conv3d_block(128, 128, (1, 2, 1), stride=(1, 2, 1), padding=(0, 0, 0)),
            conv3d_block(128, 128, (1, 2, 1), stride=(1, 2, 1), padding=(0, 0, 0)),
            conv3d_block(128, 256, (1, 4, 1), stride=(1, 2, 1), padding=(0, 0, 0)),
            # Final layer REMOVED to avoid kernel-size error!
        )
        self.fc = nn.Sequential(
            nn.Linear(256, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 1)
        )

    def forward(self, x):
        # x: [batch, n_bars, n_steps_per_bar, n_pitches, n_tracks]
        x = x.permute(0, 4, 1, 2, 3)
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
