import music21
import numpy as np
from matplotlib import pyplot as plt
import os

def binarise_output(output):
    """
    Converts MuseGAN outputs to note indices (MIDI pitches).
    Expects: [batch, bars, steps, pitches, tracks]
    Returns: [batch, bars, steps, tracks] with MIDI note indices
    """
    # Find the pitch with the highest value for each timestep and track
    max_pitches = np.argmax(output, axis=3)  # axis=3 is pitch
    return max_pitches

def notes_to_midi(output, n_bars, n_tracks, n_steps_per_bar, filename):
    """
    Converts binarized output into MIDI files, one per batch sample.
    """
    os.makedirs("./output", exist_ok=True)
    max_pitches = binarise_output(output)
    for score_num in range(len(output)):
        # Flatten bars and steps: [n_bars, n_steps_per_bar, n_tracks] -> [n_bars * n_steps_per_bar, n_tracks]
        midi_note_score = max_pitches[score_num].reshape([n_bars * n_steps_per_bar, n_tracks])
        parts = music21.stream.Score()
        parts.append(music21.tempo.MetronomeMark(number=66))
        for i in range(n_tracks):
            last_x = int(midi_note_score[:, i][0])
            s = music21.stream.Part()
            dur = 0
            for idx, x in enumerate(midi_note_score[:, i]):
                x = int(x)
                if (x != last_x or idx % 4 == 0) and idx > 0:
                    n = music21.note.Note(last_x)
                    n.duration = music21.duration.Duration(dur)
                    s.append(n)
                    dur = 0
                last_x = x
                dur += 0.25
            # Add final note
            n = music21.note.Note(last_x)
            n.duration = music21.duration.Duration(dur)
            s.append(n)
            parts.append(s)
        parts.write("midi", fp=f"./output/{filename}_{score_num}.midi")

def draw_bar(data, score_num, bar, part):
    plt.imshow(
        data[score_num, bar, :, :, part].transpose([1, 0]),
        origin="lower",
        cmap="Greys",
        vmin=-1,
        vmax=1,
    )
    plt.xlabel("Timestep")
    plt.ylabel("Pitch")
    plt.title(f"Bar {bar} - Track {part}")

def draw_score(data, score_num):
    n_bars = data.shape[1]
    n_tracks = data.shape[-1]
    fig, axes = plt.subplots(
        ncols=n_bars, nrows=n_tracks, figsize=(2*n_bars, 2*n_tracks), sharey=True, sharex=True
    )
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    for bar in range(n_bars):
        for track in range(n_tracks):
            ax = axes[track, bar] if n_bars > 1 else axes[track]
            ax.imshow(
                data[score_num, bar, :, :, track].transpose([1, 0]),
                origin="lower",
                cmap="Greys",
            )
            ax.set_title(f"Track {track}, Bar {bar}")
            ax.set_xlabel("Timestep")
            ax.set_ylabel("Pitch")
    plt.tight_layout()
    plt.show()
