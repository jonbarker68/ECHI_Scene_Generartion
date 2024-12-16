import numpy as np
import soundfile as sf
import pandas as pd
from omegaconf import OmegaConf, DictConfig
import hydra
import os
from glob import glob
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib import patches
from mutagen.flac import FLAC

np.random.seed(66)


def get_dir(base, subset="", speaker_id="", chapter=""):
    this_dir = base
    for a in [subset, speaker_id, chapter]:
        a = str(a)
        if len(a) == 0:
            break
        this_dir += f"/{a}"
    return this_dir


def read_audio(file_path, norm):
    audio, _ = sf.read(file_path)
    if norm:
        if audio.ndim == 1:
            audio = audio / np.max(np.abs(audio))
        else:
            factors = np.max(np.abs(audio), axis=1, keepdims=True)
            audio = audio / factors
    return audio


def add_audio(audio, extra_audio, start_id, channel=None):
    extra_len = extra_audio.shape[-1]
    if start_id + extra_len > audio.shape[-1]:
        audio = np.concatenate(
            [audio, np.zeros([audio.shape[0], start_id + extra_len - audio.shape[1]])],
            axis=1,
        )
    if channel is None:
        audio[:, start_id : start_id + extra_len] += extra_audio
    else:
        audio[channel, start_id : start_id + extra_len] += extra_audio
    return audio


def get_conversation(n_spk, file_options, duration):
    audio = np.zeros((n_spk, duration))
    start = 0
    old_spk = -1
    old_time = 0
    max_spk_time = 5 * 16000
    seg_lens = []
    speech_starts = []
    while start < duration:
        speaker = np.random.randint(n_spk)
        while old_spk == speaker:
            speaker = np.random.randint(n_spk)

        if start != 0:
            start = start + int(np.random.normal(loc=0, scale=500))

        new_audio = read_audio(file_options[speaker].pop(0), True)
        this_len = new_audio.shape[0]
        audio = add_audio(audio, new_audio, start, speaker)
        start += this_len
        seg_lens.append(new_audio.shape[0] / 16000)

        old_spk = speaker
    return audio, seg_lens


@hydra.main(version_base=None, config_path=".", config_name="config")
def run(cfg: DictConfig):
    libri_cfg = cfg["libri_groups"]
    tables = libri_cfg["n_channels"] // libri_cfg["spk_per_group"][-1]
    spk_per_table = libri_cfg["spk_per_group"][-1]
    convos = [[] for _ in range(tables)]
    p_switch = libri_cfg["p_switch"]

    for t in range(tables):
        for block_id in range(libri_cfg["n_blocks"]):
            if block_id == 0:
                convos[t].append(np.random.randint(2))
            else:
                switch = np.random.random() < p_switch
                if switch:
                    convos[t].append((convos[t][-1] + 1) % 2)
                else:
                    convos[t].append(convos[t][-1])

    subset = "train-clean-100"
    audio_dir = libri_cfg["root"] + "/" + subset
    libri_df = pd.read_csv(
        libri_cfg["root"] + f"/SPEAKERS.TXT",
        names=["ID", "Sex", "Subset", "Minutes", "Name"],
        sep="|",
        comment=";",
        skip_blank_lines=True,
        skipinitialspace=True,
        on_bad_lines="skip",
    )
    libri_df["Subset"] = libri_df["Subset"].map(str.strip)

    libri_df = libri_df[libri_df["Subset"] == subset]
    libri_df = libri_df[libri_df["Minutes"] > 15]

    speakers = np.random.choice(libri_df["ID"], libri_cfg["n_channels"], replace=False)

    block_duration = libri_cfg["block_size"] * libri_cfg["sample_rate"] * 60
    total_duration = block_duration * libri_cfg["n_blocks"]

    convo_signal = []
    blocks = []
    seg_lens = []

    for i, table in enumerate(convos):
        ch0, ch1 = i * spk_per_table, (i + 1) * spk_per_table
        blocks.append([])

        table_speakers = speakers[ch0:ch1]
        file_options = [
            sorted(glob(audio_dir + f"/{spk}/**/*.flac")) for spk in table_speakers
        ]

        table_audio = np.zeros([spk_per_table, total_duration])
        start = 0
        for ii, c in enumerate(table):
            if ii == 0:
                # Offset the table blocks so that they don't all split at the same time
                extra = 40 * libri_cfg["sample_rate"] * (i - 1)
            elif ii == len(table) - 1:
                # Compensate for the offset in the first frame
                extra = 40 * libri_cfg["sample_rate"] * (len(convos) - 2 - i)
            else:
                extra = 0

            if c == 0:
                # One conversation active
                new_conv, segs = get_conversation(
                    len(file_options), file_options, block_duration + extra
                )
                seg_lens += segs
            else:
                # TO DO: generate two conversations to combine
                convA, segsA = get_conversation(
                    2, file_options[:2], block_duration + extra
                )
                convB, segsB = get_conversation(
                    2, file_options[2:], block_duration + extra
                )
                seg_lens += segsA + segsB
                new_conv = np.zeros((4, max(convA.shape[-1], convB.shape[-1])))
                new_conv[:2, : convA.shape[1]] += convA
                new_conv[2:, : convB.shape[1]] += convB

            blocks[-1].append((c + 1, start, start + new_conv.shape[1]))

            new_len = new_conv.shape[-1]
            table_audio = add_audio(table_audio, new_conv, start)
            start += new_len

        convo_signal.append(table_audio)
    max_len = max(x.shape[1] for x in convo_signal)
    for i, thing in enumerate(convo_signal):
        if thing.shape[1] < max_len:
            channels = thing.shape[0]
            extra = max_len - thing.shape[1]
            convo_signal[i] = np.concatenate(
                [thing, np.zeros([channels, extra])], axis=1
            )

    convo_signal = np.concatenate(convo_signal)

    # Add noise

    demand_cfg = cfg["demand"]
    noise_dir = demand_cfg["root"] + "/PRESTO/"
    transition_time = 1
    transition_samples = transition_time * demand_cfg["sample_rate"]
    t = np.linspace(0, transition_time, demand_cfg["sample_rate"] * transition_time)
    cos = np.square(np.cos(2 * np.pi * t / (2 * transition_time)))
    sin = np.square(np.sin(2 * np.pi * t / (2 * transition_time)))

    min_length = convo_signal.shape[-1]

    file_id = 1
    mono_noise = read_audio(noise_dir + f"ch{str(file_id).zfill(2)}.wav", norm=True)
    while mono_noise.shape[-1] < min_length:
        file_id += 1
        start_len = mono_noise.shape[-1]
        signal = read_audio(noise_dir + f"ch{str(file_id).zfill(2)}.wav", norm=True)
        mono_noise[-transition_samples:] *= cos
        signal[:transition_samples] *= sin
        mono_noise = np.concat(
            [mono_noise, np.zeros(signal.shape[-1] - transition_samples)]
        )
        mono_noise[start_len - transition_samples :] += signal

    noise_signal = mono_noise[None, :min_length]
    while noise_signal.shape[0] < demand_cfg["n_channels"]:
        noise_signal = np.concatenate(
            [noise_signal, mono_noise[None, :min_length]], axis=0
        )

    full_audio = np.concatenate([convo_signal, noise_signal], axis=0)

    # save audio
    sf.write("bin/audio.wav", full_audio.T, demand_cfg["sample_rate"])

    fig, ax = plt.subplots(tables + 1, 1, figsize=[15, 8])

    xticks = [
        i for i in range(0, convo_signal.shape[-1], 60 * libri_cfg["sample_rate"])
    ]
    xticklabs = [i for i, _ in enumerate(xticks)]
    for i in range(tables):
        multi_speech = full_audio[i * spk_per_table : (i + 1) * spk_per_table]
        spks = speakers[i * spk_per_table : (i + 1) * spk_per_table]
        yticks = []
        for j, speech in enumerate(multi_speech):
            offset = spk_per_table * 2 - j * 2 - 1
            yticks.append(offset)
            ax[i].plot(speech + offset, "k", linewidth=0.2)
        ax[i].set_title(f"Table {i}")

        for c, start, end in blocks[i]:
            if c == 1:
                rect = patches.Rectangle(
                    (start, 0),
                    end - start,
                    8,
                    alpha=0.5,
                    facecolor="green",
                    edgecolor="green",
                )
                ax[i].add_patch(rect)
            else:
                colors = ["purple", "blue"]
                for ii in range(c):
                    rect = patches.Rectangle(
                        (start, ii * 4),
                        end - start,
                        4,
                        alpha=0.5,
                        facecolor=colors[ii],
                        edgecolor=colors[ii],
                    )
                    ax[i].add_patch(rect)
        ax[i].set_yticks(yticks, spks)
        ax[i].get_xaxis().set_visible(False)
    yticks = []
    for i, noise in enumerate(full_audio[-demand_cfg["n_channels"] :]):
        offset = demand_cfg["n_channels"] * 2 - i * 2 - 1
        yticks.append(offset)
        ax[-1].plot(noise + offset, "b", linewidth=0.2)

    ax[-1].set_xticks(xticks, xticklabs)
    ax[-1].set_yticks(
        yticks,
        [f"CH{libri_cfg['n_channels'] + i}" for i in range(demand_cfg["n_channels"])],
    )
    ax[-1].set_title("Noise")
    fig.supylabel("Speaker ID")
    fig.supxlabel("Time (minutes)")
    fig.suptitle("Interfering sound sources")
    plt.tight_layout()
    plt.savefig("bin/speech.png", dpi=600)
    plt.close()

    plt.hist(seg_lens)
    plt.savefig("bin/seg_lens.png", dpi=600)


def libri_analysis(cfg: DictConfig):
    libri_cfg = cfg["libri_groups"]

    files = glob(libri_cfg["root"] + "/train-clean-100/***/**/*.flac")
    file_lens = []
    for f in tqdm(files):
        with sf.SoundFile(f) as file:
            file_lens.append(len(file) / file.samplerate)
    print(np.mean(file_lens))


if __name__ == "__main__":
    # config = OmegaConf.load("config.yaml")
    # print(config)
    # run(config)
    run()
    # libri_analysis(config)
