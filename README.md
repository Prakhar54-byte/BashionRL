# BastionRL

```sh
# Install
uv sync --extra rl --dev

# Train the baseline (schema 3; retrain old checkpoints)
uv run --extra rl python -m src.game.rl train --difficulty baseline --steps 200000 --seed 0 --max-steps 2400 --output runs/baseline-v3

# Evaluate and save metrics
uv run --extra rl python -m src.game.rl evaluate --difficulty baseline --model runs/baseline-v3/final.zip --episodes 100 --seed 40000 --max-steps 2400 --output runs/baseline-v3/evaluation.json

# Short training/evaluation smoke run
uv run --extra rl python -m src.game.rl train --steps 256 --max-steps 20 --eval-freq 256 --eval-episodes 2 --output runs/smoke-v3
uv run --extra rl python -m src.game.rl evaluate --model runs/smoke-v3/final.zip --episodes 2 --max-steps 20 --output runs/smoke-v3/evaluation.json

# Human inspection and model viewing
# During manual play: 1/2/3 repair configured King Towers
uv run python main.py --difficulty baseline --seed 42
# Spectator lag check: movement should remain smooth between policy decisions
uv run --extra rl python main.py --difficulty baseline --model runs/baseline-v3/final.zip --max-steps 2400

# Optional difficulty experiments, separate from the baseline
uv run --extra rl python -m src.game.rl train --difficulty medium --steps 200000 --output runs/medium-v3
uv run --extra rl python -m src.game.rl train --difficulty hard --resume runs/medium-v3/final.zip --steps 200000 --output runs/hard-v3

# Automated checks
uv run --extra rl pytest
uv run --extra rl ruff check .
```
