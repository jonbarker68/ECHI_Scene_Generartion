# ECHI-Scene-Generation

## Installation

The package is being managed by `uv` and can be installed using the following command:

```bash
uv sync
source .venv/bin/activate
```

## Usage

The steps described below will generate the complete set of audio signals.

Note that for most scripts there are hydra configurations that are set up so that scripts will run without any arguments. However, the scripts can be run with arguments to override the default configurations if needed.

1. Set up the environment.

   ```bash
   mkdir -p data/echi_audio  # directory to store the audio signals
   ln -s /path/to/LibriSpeech data/LibriSpeech  # link to the LibriSpeech dataset
   ```

2. Build the LibriSpeech utterance index.

   ```bash
   python src/make_libri_index.py
   ```

3. Generate the master session file.

   ```bash
   python src/echi_build_master.py
   ```

4. Generate the audio signals.

   ```bash
   python src/echi_scene_renderer.py
   ```

5. Visualise the session structures (optional).

   ```bash
   python src/echi_visualiser.py session=session_001
   ```

The output will appear in a browser window.

## Extras

### Segmenting the LibriSpeech audio files

Can segment the LibriSpeech audio files into smaller segments using the following command:

```bash
python src/segment_audio.py wavfile="*.flac"
```

This will create a new directory `data/LibriSpeech_segmented/` containing the segmented audio files with the same directory structure as the original LibriSpeech dataset. You can then link to this directory instead of the original LibriSpeech directory and make a new index file, and then rebuild the master session file with the smaller segments.

## TODO

- Does not yet handle the diffuse noise speakers
