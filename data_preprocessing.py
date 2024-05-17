import os
import pretty_midi
import numpy as np
from tensorflow.keras.utils import to_categorical

# Define the path to the Lakh MIDI matched dataset
dataset_path = 'path_to_lakh_midi_matched_dataset/lmd_matched'

# Function to recursively load MIDI file paths from nested directories
def load_midi_file_paths(path):
    midi_file_paths = []
    for root, _, files in os.walk(path):
        for file in files:
            if file.endswith(".mid"):
                midi_file_paths.append(os.path.join(root, file))
    return midi_file_paths

# Function to load MIDI files from a list of file paths
def load_midi_files(file_paths):
    midi_files = []
    for file_path in file_paths:
        try:
            midi = pretty_midi.PrettyMIDI(file_path)
            midi_files.append(midi)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
    return midi_files

# Function to extract notes and durations
def extract_notes_and_durations(midi_files):
    notes = []
    durations = []
    for midi in midi_files:
        for instrument in midi.instruments:
            if not instrument.is_drum:
                for note in instrument.notes:
                    notes.append(note.pitch)
                    durations.append(note.end - note.start)
    return notes, durations

# Prepare sequences for training
def prepare_sequences(notes, sequence_length):
    sequences = []
    next_notes = []
    for i in range(0, len(notes) - sequence_length):
        sequences.append(notes[i:i + sequence_length])
        next_notes.append(notes[i + sequence_length])
    return np.array(sequences), np.array(next_notes)

# Load all MIDI file paths
all_midi_file_paths = load_midi_file_paths(dataset_path)

# Load the selected MIDI files
midi_files = load_midi_files(all_midi_file_paths)

# Extract notes and durations
notes, durations = extract_notes_and_durations(midi_files)

# Prepare sequences for training
sequence_length = 100
sequences, next_notes = prepare_sequences(notes, sequence_length)

# Normalize input data and convert to categorical
X = np.reshape(sequences, (len(sequences), sequence_length, 1))
X = X / float(len(set(notes)))  # Normalize
y = to_categorical(next_notes)

# Save preprocessed data
np.save('X.npy', X)
np.save('y.npy', y)

print("Data preprocessing complete. Saved X.npy and y.npy.")
