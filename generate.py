import torch
import music21
from utils import get_midi_note
import numpy as np

def generate_music(model, start_notes, start_durations, max_tokens, temperature, device,
                  int_to_note, int_to_duration, notes_vocab_size, durations_vocab_size):
    model.eval()
    with torch.no_grad():
        note_tokens = [int_to_note.get(n, 1) for n in start_notes]
        duration_tokens = [int_to_duration.get(d, 1) for d in start_durations]
        midi_stream = music21.stream.Stream()
        midi_stream.append(music21.clef.BassClef())
        info = []

        for note, dur in zip(start_notes, start_durations):
            new_note = get_midi_note(note, dur)
            if new_note:
                midi_stream.append(new_note)

        while len(note_tokens) < max_tokens:
            notes_tensor = torch.tensor([note_tokens], dtype=torch.long, device=device)
            durations_tensor = torch.tensor([duration_tokens], dtype=torch.long, device=device)

            note_logits, duration_logits, attn_weights = model(notes_tensor, durations_tensor)
            note_probs = torch.softmax(note_logits[0, -1] / temperature, dim=-1).cpu().numpy()
            duration_probs = torch.softmax(duration_logits[0, -1] / temperature, dim=-1).cpu().numpy()
            # Adjust for 3D attn_weights: (num_heads, num_queries, num_keys)
            attn_weights_last = attn_weights[:, -1, :len(note_tokens)].max(dim=0).values.cpu().numpy()

            sample_note_idx = np.random.choice(len(note_probs), p=note_probs)
            sample_duration_idx = np.random.choice(len(duration_probs), p=duration_probs)
            sample_note = int_to_note[sample_note_idx]
            sample_duration = int_to_duration[sample_duration_idx]
            new_note = get_midi_note(sample_note, sample_duration)

            if sample_note == "START" or len(note_tokens) >= max_tokens:
                break
            if new_note:
                midi_stream.append(new_note)
            note_tokens.append(sample_note_idx)
            duration_tokens.append(sample_duration_idx)
            info.append({
                "note_probs": note_probs,
                "atts": attn_weights_last,
                "chosen_note": (sample_note, sample_duration),
                "prompt": (note_tokens[:-1], duration_tokens[:-1])
            })

        return midi_stream, info