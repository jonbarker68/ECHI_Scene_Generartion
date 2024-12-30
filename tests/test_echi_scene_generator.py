import pandas as pd

from echi_scene_generator import (
    Speaker,
    generate_scene,
    generate_scene_node,
    make_speakers,
)


def test_generate_scene_node():
    structure = {
        "type": "sequence",
        "elements": [
            {
                "type": "conversation",
                "speakers": [1, 2],
                "duration": 10,
            }
        ],
    }
    speakers = [
        Speaker(pd.DataFrame({"speaker": [1], "file_name": ["utt1"], "length": [5]})),
        Speaker(pd.DataFrame({"speaker": [2], "file_name": ["utt2"], "length": [5]})),
    ]
    sample_rate = 1
    scene = generate_scene_node(structure, speakers, sample_rate)
    assert len(scene) == 2
    assert next(iter(scene))["type"] == "utterance"


def test_generate_scene():
    structure = {
        "type": "sequence",
        "elements": [
            {
                "type": "conversation",
                "speakers": [1, 2],
                "duration": 8,
            }
        ],
    }
    speakers = [
        Speaker(pd.DataFrame({"speaker": [1], "file_name": ["utt1"], "length": [5]})),
        Speaker(pd.DataFrame({"speaker": [2], "file_name": ["utt2"], "length": [5]})),
    ]
    sample_rate = 1
    scene = generate_scene(structure, speakers, sample_rate)
    assert len(scene) == 1
    assert scene[0]["type"] == "utterance"


def test_make_speakers():
    speech_index = pd.DataFrame(
        {
            "speaker": [1, 1, 2, 2],
            "file_name": ["utt1", "utt2", "utt3", "utt4"],
            "length": [5, 5, 5, 5],
        }
    )
    selected_speakers = [1, 2]
    speakers = make_speakers(speech_index, selected_speakers)
    assert len(speakers) == 2
    assert speakers[0].name == 1
    assert speakers[1].name == 2
