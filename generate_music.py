import argparse
import numpy as np
import pretty_midi
from tensorflow.keras.models import load_model
from midi2audio import FluidSynth

# Load the trained model
model = load_model('trained_model.h5')

def generate_music(model, seed, num_notes):
    generated = []
    for _ in range(num_notes):
        prediction = model.predict(seed)
        index = np.argmax(prediction)
        generated.append(index)
        seed = np.append(seed[:, 1:], [[[index]]], axis=1)  # Ensure index has the same dimensions
    return generated

def save_midi(generated_notes, output_path):
    midi = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=0)
    start = 0
    for note in generated_notes:
        midi_note = pretty_midi.Note(
            velocity=100, pitch=note, start=start, end=start + 0.5)
        instrument.notes.append(midi_note)
        start += 0.5
    midi.instruments.append(instrument)
    midi.write(output_path)

def midi_to_wav(midi_file, wav_file, sound_font='soundfont.sf2'):
    fs = FluidSynth(sound_font)
    fs.midi_to_audio(midi_file, wav_file)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate music using a trained model.')
    parser.add_argument('--seed', type=int, nargs='+', required=True, help='Seed sequence for the model.')
    parser.add_argument('--num_notes', type=int, default=500, help='Number of notes to generate.')
    parser.add_argument('--output', type=str, default='generated_music.mid', help='Output MIDI file.')
    parser.add_argument('--wav_output', type=str, default='generated_music.wav', help='Output WAV file.')
    parser.add_argument('--sound_font', type=str, default='soundfont.sf2', help='Path to the SoundFont file.')

    args = parser.parse_args()

    seed = np.array(args.seed).reshape((1, len(args.seed), 1))  # Reshape seed for model input
    generated_notes = generate_music(model, seed, args.num_notes)
    save_midi(generated_notes, args.output)
    
    # Convert to WAV
    midi_to_wav(args.output, args.wav_output, args.sound_font)

    print(f'Music generated and saved to {args.output} and {args.wav_output}')
