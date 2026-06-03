from il.expert import TentacleTargetFollowingExpert
from toy.rl.training import load_config
import typer
from rich.console import Console
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional

app = typer.Typer()
logger = logging.getLogger(__name__)
console = Console()
app = typer.Typer()


@app.command()
def train(config: Optional[str] = typer.Option(None, "--config", "-c")):
    cfg = load_config(config)
    expert = TentacleTargetFollowingExpert(cfg)
    print("Generating demonstrations...")
    expert.train_BC(cfg['ir']['demonstration_size'])
if __name__ == "__main__":
    app()