### README.md

# AI-Generated Music Composer

This project generates original music compositions using a trained AI model on the Lakh MIDI matched dataset. The model leverages Long Short-Term Memory (LSTM) neural networks to learn and generate sequences of musical notes.

## Project Structure

- `data_preprocessing.py`: Script for data collection and preprocessing from the entire Lakh MIDI matched dataset.
- `model_training.py`: Script for training the music generation model.
- `generate_music.py`: Script for generating music from the trained model and converting it to an audible format (WAV).
- `trained_model.h5`: Pre-trained model file (after running model_training.py).
- `requirements.txt`: List of required Python libraries.

## Requirements

- Python 3.x
- TensorFlow
- pretty_midi
- music21
- numpy
- pandas
- midi2audio
- pygame

## Installation

1. Clone the repository:

   ```sh
   git clone https://github.com/your_username/your_repository_name.git
   cd your_repository_name
   ```

2. Install the required libraries:

   ```sh
   pip install -r requirements.txt
   ```

3. Install FluidSynth using Homebrew:

   ```sh
   brew install fluid-synth
   ```

4. Download a SoundFont file (`.sf2`), for example from [here](https://sites.google.com/site/soundfonts4u/), and place it in your project directory.

## Data Preprocessing

To preprocess the data:

1. Download the Lakh MIDI matched dataset and set the `dataset_path` variable in `data_preprocessing.py` to the path where the dataset is stored.

2. Run the data preprocessing script:

   ```sh
   python data_preprocessing.py
   ```

This will generate `X.npy` and `y.npy` files, which are used for training the model. The script processes all MIDI files in the `lmd_matched` dataset to ensure comprehensive learning.

## Model Training

To train the model:

1. Ensure `X.npy` and `y.npy` files are in the project directory.

2. Run the model training script:

   ```sh
   python model_training.py
   ```

This script loads the preprocessed data and trains an LSTM-based model on the data. The model is then saved as `trained_model.h5`.

## Music Generation

To generate music using the CLI:

1. Ensure `trained_model.h5` is in the project directory.

2. Run the music generation script:

   ```sh
   python generate_music.py --seed 60 62 64 65 67 --num_notes 500 --output generated_music.mid --wav_output generated_music.wav --sound_font path_to_your_soundfont.sf2
   ```

Replace `path_to_your_soundfont.sf2` with the actual path to your SoundFont file. The script generates a MIDI file and converts it to a WAV file using FluidSynth.

### Example Usage

```sh
python generate_music.py --seed 60 62 64 65 67 --num_notes 500 --output generated_music.mid --wav_output generated_music.wav --sound_font soundfont.sf2
```

This will generate a MIDI file named `generated_music.mid` and a corresponding WAV file named `generated_music.wav`.

## License and Attribution

The Lakh MIDI Dataset is distributed with a CC-BY 4.0 license; 
Colin Raffel. "Learning-Based Methods for Comparing Sequences, with Applications to Audio-to-MIDI Alignment and Matching". PhD Thesis, 2016.

Thierry Bertin-Mahieux, Daniel P. W. Ellis, Brian Whitman, and Paul Lamere. "The Million Song Dataset". In Proceedings of the 12th International Society for Music Information Retrieval Conference, pages 591–596, 2011.

This project is licensed under the MIT License.
