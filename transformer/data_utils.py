import os
import pickle as pkl
import music21
from fractions import Fraction
import torch
from torch.utils.data import Dataset, DataLoader


def parse_midi_files(file_list, parser, seq_len, parsed_data_path=None):
    notes_list = []
    duration_list = []
    notes = []
    durations = []

    for i, file in enumerate(file_list):
        print(i + 1, "Parsing %s" % file)
        score = parser.parse(file).chordify()

        notes.append("START")
        durations.append("0.0")

        for element in score.flat:
            note_name = None
            duration_name = None

            if isinstance(element, music21.key.Key):
                note_name = str(element.tonic.name) + ":" + str(element.mode)
                duration_name = "0.0"
            elif isinstance(element, music21.meter.TimeSignature):
                note_name = str(element.ratioString) + "TS"
                duration_name = "0.0"
            elif isinstance(element, music21.chord.Chord):
                note_name = element.pitches[-1].nameWithOctave
                duration_name = str(element.duration.quarterLength)
            elif isinstance(element, music21.note.Rest):
                note_name = str(element.name)
                duration_name = str(element.duration.quarterLength)
            elif isinstance(element, music21.note.Note):
                note_name = str(element.nameWithOctave)
                duration_name = str(element.duration.quarterLength)

            if note_name and duration_name:
                notes.append(note_name)
                durations.append(duration_name)

    print(f"Building sequences of length {seq_len}")
    for i in range(len(notes) - seq_len):
        notes_list.append(notes[i: i + seq_len + 1])
        duration_list.append(durations[i: i + seq_len + 1])

    if parsed_data_path:
        os.makedirs(parsed_data_path, exist_ok=True)
        with open(os.path.join(parsed_data_path, "notes.pkl"), "wb") as f:
            pkl.dump(notes_list, f)
        with open(os.path.join(parsed_data_path, "durations.pkl"), "wb") as f:
            pkl.dump(duration_list, f)

    return notes_list, duration_list


def load_parsed_files(parsed_data_path):
    with open(os.path.join(parsed_data_path, "notes.pkl"), "rb") as f:
        notes = pkl.load(f)
    with open(os.path.join(parsed_data_path, "durations.pkl"), "rb") as f:
        durations = pkl.load(f)
    return notes, durations


def build_vocabulary(sequences):
    unique_items = set()
    for seq in sequences:
        unique_items.update(seq)
    vocab = {item: idx for idx, item in enumerate(sorted(unique_items))}
    return vocab


class MusicDataset(Dataset):
    def __init__(self, notes_sequences, durations_sequences, note_to_int, duration_to_int):
        self.notes_sequences = notes_sequences
        self.durations_sequences = durations_sequences
        self.note_to_int = note_to_int
        self.duration_to_int = duration_to_int

    def __len__(self):
        return len(self.notes_sequences)

    def __getitem__(self, idx):
        notes_seq = self.notes_sequences[idx]
        durations_seq = self.durations_sequences[idx]
        notes_input = [self.note_to_int[note] for note in notes_seq[:-1]]
        notes_target = [self.note_to_int[note] for note in notes_seq[1:]]
        durations_input = [self.duration_to_int[d] for d in durations_seq[:-1]]
        durations_target = [self.duration_to_int[d] for d in durations_seq[1:]]
        return (
            torch.tensor(notes_input, dtype=torch.long),
            torch.tensor(durations_input, dtype=torch.long),
            torch.tensor(notes_target, dtype=torch.long),
            torch.tensor(durations_target, dtype=torch.long)
        )


def prepare_data(file_list, seq_len, batch_size, parsed_data_path=None, parse_midi=True):
    parser = music21.converter
    if parse_midi:
        notes_seqs, durations_seqs = parse_midi_files(file_list, parser, seq_len, parsed_data_path)
    else:
        notes_seqs, durations_seqs = load_parsed_files(parsed_data_path)

    note_to_int = build_vocabulary(notes_seqs)
    duration_to_int = build_vocabulary(durations_seqs)
    int_to_note = {idx: note for note, idx in note_to_int.items()}
    int_to_duration = {idx: dur for dur, idx in duration_to_int.items()}

    dataset = MusicDataset(notes_seqs, durations_seqs, note_to_int, duration_to_int)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    return dataloader, note_to_int, duration_to_int, int_to_note, int_to_duration