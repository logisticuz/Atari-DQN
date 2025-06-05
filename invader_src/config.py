import argparse
# denna fil innehåller konfigurationsinställningar för träningsskriptet
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--infinite", action="store_true")
    parser.add_argument("--reset-buffer", action="store_true")
    parser.add_argument("--episodes", type=int, default=1_000_000)
    parser.add_argument("--logdir", type=str, default="logs")
    parser.add_argument("--reward-shaping", action="store_true")
    parser.add_argument("--tensorboard", action="store_true")
    parser.add_argument("--resume-step", type=int)
    parser.add_argument("--no-prioritized", dest="prioritized", action="store_false", help="Disable PER")
    parser.set_defaults(prioritized=True)
    parser.add_argument("--overlay", action="store_true", help="Enable overlays during training")

    return parser.parse_args()
