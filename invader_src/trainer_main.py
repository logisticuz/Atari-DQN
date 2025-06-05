import argparse
import subprocess
from rich.console import Console
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import parse_args
from utils.reward_config import REWARD_CONFIG
from utils.video_logger import VideoLogger
from trainer_loop import train_loop

console = Console()

def launch_tensorboard(logdir: str):
    console.print(f"[bold blue]📊 Startar TensorBoard på logdir:[/bold blue] {logdir}")
    subprocess.Popen(["tensorboard", "--logdir", logdir])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run different RL modes")
    parser.add_argument("--mode", type=str, choices=["train"], required=True, help="Which mode to run")
    parser.add_argument("--episodes", type=int, default=500_000, help="Number of steps to train")
    parser.add_argument("--reward-shaping", action="store_true", help="Use reward shaping")
    parser.add_argument("--resume-step", type=int, help="Resume from saved step")
    parser.add_argument("--reset-buffer", action="store_true", help="Ignore saved replay buffer")
    parser.add_argument("--infinite", action="store_true", help="Train indefinitely")
    parser.add_argument("--tensorboard", action="store_true", help="Launch TensorBoard")
    parser.add_argument("--logdir", type=str, default="logs", help="TensorBoard log directory")
    parser.add_argument("--force-left", action="store_true", help="Force agent to explore left movement initially.")
    
    # Lägg till overlay-flagga
    parser.add_argument("--overlay", action="store_true", help="Enable overlays during training")

    args = parser.parse_args()

    # Starta TensorBoard om det efterfrågas
    if args.tensorboard:
        launch_tensorboard(args.logdir)

    # Skapa overlay_flags baserat på --overlay-argumentet
    overlay_flags = {}
    if args.overlay:
        overlay_flags = {
            "zone_boxes": True,
            "troop_boxes": True,
            "range_label": True,
            "shelter_highlight": True
        }

    # Kör träningsloopen
    if args.mode == "train":
        train_loop(args, overlay_flags) 
    else:
        console.print("[red]❌ Unsupported mode[/red]")
