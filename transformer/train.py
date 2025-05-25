import os
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import music21
import numpy as np
from utils import get_midi_note

def train_model(
    model,
    train_loader,
    val_loader,
    epochs,
    device,
    notes_vocab_size,
    durations_vocab_size,
    generate_len,
    output_dir,
    note_to_int,
    duration_to_int,
    int_to_note,
    int_to_duration,
    patience: int = 50
):
    os.makedirs(output_dir, exist_ok=True)

    optimizer = Adam(model.parameters(), lr=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)
    crit_n = nn.CrossEntropyLoss()
    crit_d = nn.CrossEntropyLoss()

    best_val_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(1, epochs + 1):
        # ——— TRAINING STEP ———
        model.train()
        total_train = 0.0
        for notes_in, durs_in, notes_tgt, durs_tgt in train_loader:
            notes_in, durs_in   = notes_in.to(device),   durs_in.to(device)
            notes_tgt, durs_tgt = notes_tgt.to(device), durs_tgt.to(device)

            optimizer.zero_grad()
            logits_n, logits_d, _ = model(notes_in, durs_in)

            loss_n = crit_n(logits_n.view(-1, notes_vocab_size),     notes_tgt.view(-1))
            loss_d = crit_d(logits_d.view(-1, durations_vocab_size), durs_tgt.view(-1))
            loss   = loss_n + loss_d

            loss.backward()
            optimizer.step()
            total_train += loss.item()

        avg_train = total_train / len(train_loader)

        # ——— VALIDATION STEP ———
        model.eval()
        total_val = 0.0
        with torch.no_grad():
            for notes_in, durs_in, notes_tgt, durs_tgt in val_loader:
                notes_in, durs_in   = notes_in.to(device),   durs_in.to(device)
                notes_tgt, durs_tgt = notes_tgt.to(device), durs_tgt.to(device)
                logits_n, logits_d, _ = model(notes_in, durs_in)
                loss_n = crit_n(logits_n.view(-1, notes_vocab_size),     notes_tgt.view(-1))
                loss_d = crit_d(logits_d.view(-1, durations_vocab_size), durs_tgt.view(-1))
                total_val += (loss_n + loss_d).item()

        avg_val = total_val / len(val_loader)
        scheduler.step(avg_val)

        print(f"Epoch {epoch}/{epochs} — Train: {avg_train:.4f} — Val: {avg_val:.4f}")

        # ——— CHECKPOINT CALLBACK ———
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            epochs_no_improve = 0
            torch.save(model.state_dict(), os.path.join(output_dir, "best_model.pt"))
            print(f"  ↳ New best model (val_loss={best_val_loss:.4f}) saved.")
        else:
            epochs_no_improve += 1

        # ——— EARLY STOPPING CALLBACK ———
        if epochs_no_improve >= patience:
            print(f"Stopping early (no improvement for {patience} epochs).")
            break

        # ——— OPTIONAL SAMPLE CALLBACK ———
        if epoch % 50 == 0:
            midi_stream, _ = generate_music(
                model,
                start_notes     = ["START"],
                start_durations = ["0.0"],
                max_tokens      = generate_len,
                temperature     = 0.5,
                device          = device,
                note_to_int     = note_to_int,
                duration_to_int = duration_to_int,
                int_to_note     = int_to_note,
                int_to_duration = int_to_duration,
            )
            midi_stream.write("midi", fp=os.path.join(output_dir, f"sample_epoch_{epoch:04d}.mid"))

    print("Training complete. Best val loss:", best_val_loss)


def generate_music(
    model,
    start_notes,
    start_durations,
    max_tokens,
    temperature,
    device,
    note_to_int,
    duration_to_int,
    int_to_note,
    int_to_duration
):
    """
    Returns: (music21.stream.Stream, info_list)
      info_list = [
        {
          "note_probs": np.ndarray,
          "atts":       np.ndarray,   # shape = (P,) for each step
          "chosen_note": (str, str),
          "prompt":      (List[int], List[int])
        }, ...
      ]
    """
    model.eval()
    info = []
    midi_stream = music21.stream.Stream()
    midi_stream.append(music21.clef.BassClef())

    # -- seed tokens & stream --
    note_tokens     = [note_to_int.get(n, note_to_int["START"]) for n in start_notes]
    duration_tokens = [duration_to_int.get(d, duration_to_int["0.0"]) for d in start_durations]
    for n, d in zip(start_notes, start_durations):
        nnote = get_midi_note(n, d)
        if nnote: midi_stream.append(nnote)

    with torch.no_grad():
        while len(note_tokens) < max_tokens:
            nt = torch.tensor([note_tokens],     device=device)
            dt = torch.tensor([duration_tokens], device=device)

            logits_n, logits_d, aw = model(nt, dt)

            probs_n = torch.softmax(logits_n[0, -1] / temperature,    dim=-1).cpu().numpy()
            probs_d = torch.softmax(logits_d[0, -1] / temperature,    dim=-1).cpu().numpy()

            # — robust attention handling —
            aw_cpu = aw.detach().cpu()
            if aw_cpu.ndim == 3:
                # (heads, Q, K)
                last_att = aw_cpu[:, -1, :len(note_tokens)].max(dim=0).values
            elif aw_cpu.ndim == 2:
                # (Q, K)
                last_att = aw_cpu[-1, :len(note_tokens)]
            else:
                raise RuntimeError(f"Unexpected aw.ndim={aw_cpu.ndim}")
            attn_last = last_att.numpy()

            # sample next token
            idx_n = np.random.choice(len(probs_n), p=probs_n)
            idx_d = np.random.choice(len(probs_d), p=probs_d)
            s_n = int_to_note[idx_n]
            s_d = int_to_duration[idx_d]

            if s_n == "START":
                break

            midi_note = get_midi_note(s_n, s_d)
            if midi_note:
                midi_stream.append(midi_note)

            # update prompt & record info
            note_tokens.append(idx_n)
            duration_tokens.append(idx_d)
            info.append({
                "note_probs":    probs_n,
                "atts":          attn_last,
                "chosen_note":   (s_n, s_d),
                "prompt":        (note_tokens[:-1], duration_tokens[:-1])
            })

    return midi_stream, info
