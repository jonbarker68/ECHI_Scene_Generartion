"""Script to visualise sessions from the ECHI master file using plotly.

usage: python echi_visualiser.py +session=SESSION_NAME
e.g. python src/echi_visualiser.py +session=session_001
"""

import json

import hydra
import numpy as np
import plotly.graph_objects as go  # type: ignore


def get_element_duration(element):
    """Get the duration of an element."""
    if element["type"] == "splitter":
        element = element["elements"][0]
    return element["duration"]


def make_plot(session_dict, fig):
    """Make a plot of the session structure."""

    # Construct the conversation structure segment sequence
    sample_rate = session_dict["sample_rate"]
    splits = session_dict["structure"]["elements"][0]["elements"]
    segments = []
    for sequences in splits:
        speakers = sequences["speakers"]
        seg_types = [element["type"] for element in sequences["elements"]]
        durations = [get_element_duration(element) for element in sequences["elements"]]
        end_times = np.cumsum(durations)
        start_times = np.concatenate(([0], end_times[:-1]))
        for seg_type, start_time, end_time in zip(seg_types, start_times, end_times):
            segments.append(
                {
                    "start": start_time,
                    "end": end_time,
                    "speakers": speakers,
                    "seg_type": seg_type,
                }
            )

    # Plot the elements to illustrate the conversation structure
    for event in segments:
        speaker_min = min(event["speakers"])
        speaker_max = max(event["speakers"])
        color = "blue" if event["seg_type"] == "conversation" else "green"
        name = (
            "One conversation"
            if event["seg_type"] == "conversation"
            else "Split conversation"
        )
        fig.add_trace(
            go.Scatter(
                x=[
                    event["start"],
                    event["end"],
                    event["end"],
                    event["start"],
                    event["start"],
                ],
                y=[
                    speaker_min - 0.4,
                    speaker_min - 0.4,
                    speaker_max + 0.4,
                    speaker_max + 0.4,
                    speaker_min - 0.4,
                ],
                fill="toself",
                line=dict(color=color),
                name=name,
                hoverinfo="text",
                text=name,
                showlegend=False,  # Disable legend for this trace
            )
        )

    # Add elements to illustrate the individual utterances
    for event in session_dict["scene"]:
        channel = event["channel"]
        start_time = event["onset"] / sample_rate
        end_time = event["offset"] / sample_rate
        fig.add_trace(
            go.Scatter(
                x=[start_time, end_time],
                y=[channel, channel],
                mode="lines",
                line=dict(color="red", width=10),
                name=event["type"],
                hoverinfo="text",
                text=event["filename"],
                showlegend=False,  # Disable legend for this trace
            )
        )

    # Draw the figure
    fig.update_layout(
        title=f"Session {session_dict['session']}",
        xaxis_title="Time",
        yaxis=dict(
            title="Speaker",
            tickmode="linear",
            tick0=1,
            dtick=1,
        ),
        xaxis=dict(
            title="Time",
            tickmode="linear",
            tick0=0,
            dtick=300,
        ),
        height=800,
        width=2000,
        showlegend=False,  # Disable legend for the entire figure
    )

    return fig


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg):
    """Visualise the ECHI master file."""

    with open(cfg.paths.master_file, "r", encoding="utf8") as f:
        master = json.load(f)

    # find the session called cfg.session
    session = next(
        (
            session_dict
            for session_dict in master
            if session_dict["session"] == cfg.session
        ),
        None,
    )

    if not session:
        raise ValueError(f"Session {cfg.session} not found in master file.")

    fig = go.Figure()

    make_plot(session, fig)

    fig.show()


if __name__ == "__main__":
    main()
