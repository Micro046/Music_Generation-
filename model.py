import torch
import torch.nn as nn
import numpy as np

class SinePositionEncoding(nn.Module):
    def __init__(self, max_wavelength=10000):
        super(SinePositionEncoding, self).__init__()
        self.max_wavelength = max_wavelength

    def forward(self, x):
        seq_len, embed_dim = x.shape[1], x.shape[2]
        position = torch.arange(seq_len, dtype=torch.float, device=x.device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2, dtype=torch.float, device=x.device) * (-np.log(self.max_wavelength) / embed_dim))
        pe = torch.zeros(1, seq_len, embed_dim, device=x.device)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        return x + pe

class TokenAndPositionEmbedding(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super(TokenAndPositionEmbedding, self).__init__()
        self.token_emb = nn.Embedding(vocab_size, embed_dim)
        self.pos_enc = SinePositionEncoding()

    def forward(self, x):
        emb = self.token_emb(x)
        return self.pos_enc(emb)

class TransformerBlock(nn.Module):
    def __init__(self, num_heads, key_dim, embed_dim, ff_dim, dropout_rate=0.3):
        super(TransformerBlock, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout_rate)  # Remove kdim, vdim
        self.dropout1 = nn.Dropout(dropout_rate)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_dim)
        )
        self.dropout2 = nn.Dropout(dropout_rate)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        seq_len, batch_size, _ = x.shape
        mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
        attn_output, attn_weights = self.attention(x, x, x, attn_mask=mask, need_weights=True)
        x = self.norm1(x + self.dropout1(attn_output))
        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout2(ffn_output))
        return x, attn_weights

class MusicTransformer(nn.Module):
    def __init__(self, notes_vocab_size, durations_vocab_size, embed_dim, num_heads, key_dim, ff_dim, dropout_rate=0.3):
        super(MusicTransformer, self).__init__()
        self.note_embedding = TokenAndPositionEmbedding(notes_vocab_size, embed_dim // 2)
        self.duration_embedding = TokenAndPositionEmbedding(durations_vocab_size, embed_dim // 2)
        self.transformer_block = TransformerBlock(num_heads, key_dim, embed_dim, ff_dim, dropout_rate)
        self.note_output = nn.Linear(embed_dim, notes_vocab_size)
        self.duration_output = nn.Linear(embed_dim, durations_vocab_size)

    def forward(self, notes, durations):
        note_emb = self.note_embedding(notes)
        duration_emb = self.duration_embedding(durations)
        emb = torch.cat((note_emb, duration_emb), dim=-1)
        emb = emb.permute(1, 0, 2)  # (seq_len, batch_size, embed_dim)
        transformer_output, attn_weights = self.transformer_block(emb)
        transformer_output = transformer_output.permute(1, 0, 2)  # (batch_size, seq_len, embed_dim)
        note_logits = self.note_output(transformer_output)
        duration_logits = self.duration_output(transformer_output)
        return note_logits, duration_logits, attn_weights