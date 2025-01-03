# ECHI-Scene-Generation

## Installation

The package is being managed by `uv` and can be installed using the following command:

```bash
uv sync
source .venv/bin/activate
```

## Usage

The steps described below will generate the complete set of audio signals.

Note that for all scripts there are hydra configurations that are set up so that scripts will run without any arguments. However, the scripts can be run with arguments to override the default configurations if needed.

1. Set up the environment.

   ```bash
   mkdir -p data/echi_audio  # directory to store the audio signals
   ln -s /path/to/LibriSpeech data/LibriSpeech  # link to the LibriSpeech dataset
   ```

2. Segment the LibriSpeech utterances

   ```bash
   python src/segment_audio.py
   # Redirect data/LibriSpeech to the segmented data
   rm data/LibriSpeech
   ln -s LibriSpeech_segmented data/LibriSpeech
   ```

3. Build the LibriSpeech utterance index.

   ```bash
   python src/make_libri_index.py
   ```

4. Generate the master session file.

   ```bash
   python src/echi_build_master.py
   ```

5. Generate the audio signals.

   ```bash
   python src/echi_scene_renderer.py
   ```

6. Visualise the session structures (optional).

   ```bash
   python src/echi_visualiser.py session=session_001
   ```

The output will appear in a browser window.
