# BastionRL

```sh
# Install
uv sync --extra rl --dev

# Train on medium
uv run --extra rl python -m src.game.rl train --difficulty medium --steps 200000 --output runs/medium

# Continue training on hard
uv run --extra rl python -m src.game.rl train --difficulty hard --resume runs/medium/final.zip --steps 200000 --output runs/hard

# Test the medium model on both difficulties
uv run --extra rl python -m src.game.rl evaluate --difficulty medium --model runs/medium/final.zip --episodes 100 --seed 40000
uv run --extra rl python -m src.game.rl evaluate --difficulty hard --model runs/medium/final.zip --episodes 100 --seed 40000

# Test the hard model
uv run --extra rl python -m src.game.rl evaluate --difficulty hard --model runs/hard/final.zip --episodes 100 --seed 40000

# Human testing
uv run python main.py --difficulty medium --seed 42
uv run python main.py --difficulty hard --seed 42

# Watch the trained model
uv run --extra rl python main.py --difficulty hard --model runs/hard/final.zip

# Automated checks
uv run --extra rl pytest
uv run --extra rl ruff check .
```
