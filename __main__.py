import sys
from rl.training import train
from rl.loaders import load_rl_config
def main(config_path: str = None) -> None:
    train(config_path)


if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else None
    main(config_path)