import argparse
import glob
from pathlib import Path

# Compatibility shim for old NumPy versions where numpy.typing.NDArray is absent.
import numpy.typing as npt

if not hasattr(npt, "NDArray"):
    class _FakeNDArray:
        def __class_getitem__(cls, _item):
            return object

    npt.NDArray = _FakeNDArray

import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator


def plot_all_in_one_window(
    reward_x,
    reward_y,
    success_x,
    success_y,
    episode_length_x,
    episode_length_y,
):
    fig, axes = plt.subplots(3, 1, figsize=(10, 12))

    axes[0].plot(reward_x, reward_y)
    axes[0].set_title("Mean Reward Per Episode")
    axes[0].set_xlabel("step")
    axes[0].set_ylabel("mean reward per episode")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(success_x, success_y)
    axes[1].set_title("Episode Success Rate")
    axes[1].set_xlabel("step")
    axes[1].set_ylabel("success rate")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(episode_length_x, episode_length_y)
    axes[2].set_title("Mean Episode Length")
    axes[2].set_xlabel("step")
    axes[2].set_ylabel("mean episode length")
    axes[2].grid(True, alpha=0.3)

    fig.tight_layout()


def main():
    parser = argparse.ArgumentParser(description="Plot scalar TensorBoard logs for a selected run.")
    parser.add_argument(
        "--run",
        required=True,
        help="Run path inside unitree_rl_gym/logs, for example: rough_go2/Apr23_13-21-35_",
    )
    parser.add_argument(
        "--single-window",
        action="store_true",
        help="Show all graphs in one window.",
    )
    args = parser.parse_args()

    logs_root = Path(__file__).resolve().parents[2] / "logs"
    run_dir = logs_root / args.run
    event_files = sorted(glob.glob(str(run_dir / "events.out.tfevents.*")))

    if not event_files:
        raise FileNotFoundError(f"No TensorBoard event files found in: {run_dir}")

    accumulator = event_accumulator.EventAccumulator(event_files[-1])
    accumulator.Reload()
    scalar_tags = set(accumulator.Tags().get("scalars", []))
    reward_tag_candidates = [
        "Train/mean_reward",
        "Episode/reward",
        "Episode/mean_reward",
    ]
    reward_tag = next((tag for tag in reward_tag_candidates if tag in scalar_tags), None)
    success_rate_tag = "Episode/success_rate"
    episode_length_tag_candidates = [
        "Train/mean_episode_length",
        "Episode/mean_episode_length",
        "Episode/episode_length",
    ]
    episode_length_tag = next(
        (tag for tag in episode_length_tag_candidates if tag in scalar_tags), None
    )

    if reward_tag is None:
        raise RuntimeError(
            f"Reward tag not found. Available scalar tags: {sorted(scalar_tags)}"
        )
    if success_rate_tag not in scalar_tags:
        raise RuntimeError(
            f"{success_rate_tag} not found. Available scalar tags: {sorted(scalar_tags)}"
        )
    if episode_length_tag is None:
        raise RuntimeError(
            f"Episode length tag not found. Available scalar tags: {sorted(scalar_tags)}"
        )

    events = accumulator.Scalars(reward_tag)
    x = [event.step for event in events]
    y = [event.value for event in events]

    success_events = accumulator.Scalars(success_rate_tag)
    success_x = [event.step for event in success_events]
    success_y = [event.value for event in success_events]

    episode_length_events = accumulator.Scalars(episode_length_tag)
    episode_length_x = [event.step for event in episode_length_events]
    episode_length_y = [event.value for event in episode_length_events]

    if args.single_window:
        plot_all_in_one_window(
            x,
            y,
            success_x,
            success_y,
            episode_length_x,
            episode_length_y,
        )
    else:
        plt.figure(figsize=(10, 5))
        plt.plot(x, y)
        plt.title("Mean Reward Per Episode")
        plt.xlabel("step")
        plt.ylabel("mean reward per episode")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        plt.figure(figsize=(10, 5))
        plt.plot(success_x, success_y)
        plt.title("Episode Success Rate")
        plt.xlabel("step")
        plt.ylabel("success rate")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        plt.figure(figsize=(10, 5))
        plt.plot(episode_length_x, episode_length_y)
        plt.title("Mean Episode Length")
        plt.xlabel("step")
        plt.ylabel("mean episode length")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()
