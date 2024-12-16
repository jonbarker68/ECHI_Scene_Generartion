from omegaconf import OmegaConf
from argparse import ArgumentParser
import os
from soundfile import read, write
import numpy as np

CONFIG = OmegaConf.load("../config.yaml")

if __name__ == "__main__":
    args_parser = ArgumentParser()
    args_parser.add_argument('-F', '--file_path')
    args_parser.add_argument('-T', '--close_talk')
    args_parser.add_argument('-C', '--channels')
    args = args_parser.parse_args()
    
    file_path = os.path.join(CONFIG.data_root, args.file_path)
    if os.path.exists(file_path):
        audio, fs = read(file_path)
    else:
        raise ValueError(f"No file found at:\n{file_path}")

    close_talk = int(args.close_talk)
    if "," in args.channels:
        channels = [int(s) for s in args.channels.split(",")]
    elif args.channels == "all":
        channels = [i for i in range(audio.shape[-1])]
    else:
        raise ValueError(f"Channels arg {args.channels} not recognised!\nWhat do you want from meeeeeeeee???????")
    
    assert close_talk < audio.shape[-1]

    test_file = file_path.split("/")[-1][:-4]
    store = {"close_talk": audio[:, close_talk]}
    store = {"close_talk_short": audio[:fs * 120, close_talk]}
    for ch in channels:
        store[f"ch{ch}"] = audio[:, ch]
    
    for name, chaudio in store.items():
        new_file = file_path[:-4] + f"_{name}.wav"
        write(new_file, chaudio, fs)