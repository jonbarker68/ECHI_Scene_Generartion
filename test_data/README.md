# Test data

A collection of small test files packaged with the code to allow easy testing.

## echi_scene.json

An example of the scene file format that can be used to test the scene renderer.

- The scene file is a list of dictionaries, each dictionary representing a single audio
segment.
- The segments can be 'file' or 'generator'.
- File segments have a start time, end time, wav file path, and channel number.
- Generator segments have a start time, end time and generator parameters (TBD).
- Filenames are all relative to a common root specified in the config.

TODO: Need to consider how to handle things if files can come from different datasets.
May need to have a way to specify separate roots, or install all datasets under a
common root with symlinks and then have longer paths in the scene file.
