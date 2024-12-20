# ECHI Scene Generation

## 1. Overview

Scene generation is a multi-stage process

```mermaid
---
config:
  theme: dark
---
flowchart TD
    A[echi.config] -- structure_generator --> B[structure.json]
    B -- scene_generator --> C[scene.json]
    C -- scene_renderer -->DC[audio.wav]
```

## 2. Scene Rendering

Turns a scene file into a multichannel audio file.

### 2.1 Scene File Format

The scene file is a list of dictionaries, each dictionary representing a single audio segment.

### 2.2 Scene Renderer

The scene render will generate multichannel audio from the scene file. This is a straightforward process of pre-allocating an n-channels by n-samples array and then filling in the samples from the audio segments.

## 3. Scene Generation

Scene generation will make a low level scene file from the high level structure file.

### 3.1 structure File Format

Scenes are described in a nested dictionary format. With objects to represent sequences, splitters, conversations, noises and pauses.

Example structure file:

```json
{
    "type": "sequence",
    "speakers": [1,2,3,4,5,6,7,8,9,10,11,12],
    "elements": [
       {
        "type": "splitter",
        "elements": [
            {"type": "sequence",
             "speakers": [1,2,3,4],
             "elements": [
                    {"type": "pause",
                     "duration": 20
                    },
                    {"type": "conversation",
                     "speakers": [1,2,3],
                     "duration": 120
                    },
                    {"type": "conversation",
                     "speakers": [1,3,4],
                     "duration": 120
                    },
                    {"type": "conversation",
                     "speakers": [3,4],
                     "duration": 120
                    }
                ]
            },
            {"type": "sequence",
             "speakers": [5,6,7,8],
             "elements":  [
                    {"type": "conversation",
                     "speakers": [5,6,7,8],
                     "duration": 120
                    },
                    {"type": "splitter",
                    "elements": [
                            {"type": "conversation",
                            "speakers": [5,6],
                            "duration": 120
                            },
                            {"type": "conversation",
                            "speakers": [7,8],
                            "duration": 120
                            }
                        ]
                    },
                    {"type": "conversation",
                     "speakers": [5,6,7,8],
                     "duration": 120
                    }
                ]
            },
            {"type": "conversation",
             "speakers": [9,10,11,12],
             "duration": 360
            }
        ]
       }
    ]
}
```

### 3.2 Scene Generator

This is a fairly complex process that involves instantiating a structure to form a low level `scene.json` file. This is done by recursively walking the structure and instantiating each object into a sequence of audio segments. The conversation objects are constructed using rules that govern turn taking and overlap. In sequence, objects are rendered sequentially in time. For splitter objects, each object in the splitter is rendered in parallel, and the object is not considered finished until all objects in the splitter are finished.
