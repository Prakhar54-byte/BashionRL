# BastionRL

```sh
# Install
uv sync --extra rl --dev

# Fresh schema-7 training after upstream sync (older checkpoints require the old code)
uv run --extra rl python -m src.game.rl train --difficulty baseline --strategy hierarchical --steps 2000000 --seed 0 --n-envs 8 --n-steps 512 --batch-size 256 --n-epochs 5 --learning-rate 0.0003 --lr-final 0.00005 --ent-coef 0.02 --ent-final 0.005 --curriculum mixed --promotion-successes 30 --normalize --eval-freq 65536 --eval-episodes 30 --output runs/mixed-v3-main-rebuild/seed-0

# Fresh CUDA server run using the existing, successfully installed GPU environment
# Use an unused output directory; do not restart an active run just to update this guide
../bashion-cuda-env/bin/python -c "import pygame, gymnasium, sb3_contrib, torch; assert torch.cuda.is_available(); x=torch.ones(2,2,device='cuda'); print(torch.cuda.get_device_name(0), (x @ x).sum().item())"
server_output="runs/mixed-v3-main-rebuild-gpu/seed-0"
mkdir -p "$server_output"
nohup ../bashion-cuda-env/bin/python -u -m src.game.rl train \
  --difficulty baseline --strategy hierarchical --steps 2000000 --seed 0 --device cuda \
  --n-envs 8 --n-steps 512 --batch-size 256 --n-epochs 5 \
  --learning-rate 0.0003 --lr-final 0.00005 --ent-coef 0.02 --ent-final 0.005 \
  --curriculum mixed --promotion-successes 30 --normalize \
  --max-steps 2400 --eval-freq 65536 --eval-episodes 30 \
  --output "$server_output" > "$server_output/training.log" 2>&1 < /dev/null &
echo $! > "$server_output/training.pid"
tail -F "$server_output/training.log"

# GPU-run evaluation (uses frozen normalization statistics; evaluation runs on CPU)
../bashion-cuda-env/bin/python -m src.game.rl evaluate \
  --difficulty baseline --model "$server_output/best_model.zip" \
  --episodes 100 --seed 90000 --max-steps 2400 --output "$server_output/evaluation.json"

# Inspect throughput of the current server run without restarting it
grep -E 'fps[[:space:]]|total_timesteps|time_elapsed' runs/mixed-v3-main-rebuild/seed-0/training.log | tail -12
# Approximate remaining hours = (2000000 - completed training steps) / (fps * 3600)

# Evaluate raw game rewards with frozen checkpoint normalization statistics
uv run --extra rl python -m src.game.rl evaluate --model runs/mixed-v3-main-rebuild/seed-0/best_model.zip --episodes 100 --seed 90000 --output runs/mixed-v3-main-rebuild/seed-0/evaluation.json

# Resume the same experiment; --steps is additional (use the remaining-budget formula below for a fixed target)
uv run --extra rl python -m src.game.rl train --resume runs/mixed-v3-main-rebuild/seed-0/latest.zip --steps 1000000 --n-envs 8 --eval-freq 65536 --eval-episodes 30 --output runs/mixed-recovered

# Short training/evaluation smoke run
uv run --extra rl python -m src.game.rl train --steps 256 --max-steps 20 --eval-freq 256 --eval-episodes 2 --output runs/hierarchical-smoke
uv run --extra rl python -m src.game.rl evaluate --model runs/hierarchical-smoke/final.zip --episodes 2 --max-steps 20 --output runs/hierarchical-smoke/evaluation.json

# Human inspection and model viewing
uv run python main.py --difficulty baseline --seed 42
uv run --extra rl python main.py --difficulty baseline --model runs/mixed-v3-main-rebuild/seed-0/best_model.zip --max-steps 2400

# Optional difficulty experiments, separate from the baseline
uv run --extra rl python -m src.game.rl train --difficulty medium --steps 2000000 --output runs/hierarchical-medium
uv run --extra rl python -m src.game.rl train --difficulty hard --resume runs/hierarchical-medium/final.zip --steps 2000000 --output runs/hierarchical-hard

# Scripted winnability probes (separate from RL; no extra health/resources)
uv run --extra rl python -m src.game.rl diagnose --episodes 10 --seed 80000 --output runs/winnability-defense.json
uv run --extra rl python -m src.game.rl diagnose --economy --episodes 10 --seed 80000 --output runs/winnability-economy.json

# Matched rollout examples: 1 x 1024 and 4 x 256 both collect 1024 samples/update
uv run --extra rl python -m src.game.rl train --strategy flat --n-envs 1 --n-steps 1024 --batch-size 64 --n-epochs 10 --steps 2000000 --seed 0 --eval-freq 65536 --eval-episodes 30 --output runs/matched-flat
uv run --extra rl python -m src.game.rl train --strategy hierarchical --n-envs 4 --n-steps 256 --batch-size 64 --n-epochs 10 --steps 2000000 --seed 0 --eval-freq 65536 --eval-episodes 30 --output runs/matched-hierarchical

# Server comparison: two policies x three seeds, run sequentially; requires capacity for 16 workers
# Every case: rollout=4096, batch=64, epochs<=10, same validation schedule and budget
mkdir -p runs/matched-2m
nohup bash -c '
set -eu
for training_seed in 0 1 2; do
  for strategy in flat hierarchical; do
    output="runs/matched-2m/${strategy}-seed-${training_seed}"
    mkdir -p "$output"
    uv run --extra rl python -u -m src.game.rl train \
      --strategy "$strategy" --steps 2000000 --seed "$training_seed" \
      --n-envs 16 --n-steps 256 --batch-size 64 --n-epochs 10 --device cpu \
      --max-steps 2400 --eval-freq 65536 --eval-episodes 30 --eval-seed 10000 \
      --output "$output" > "$output/training.log" 2>&1
    uv run --extra rl python -m src.game.rl evaluate \
      --model "$output/best_model.zip" --episodes 100 --seed 90000 \
      --max-steps 2400 --output "$output/evaluation.json" > "$output/evaluation.log" 2>&1
  done
done
' > runs/matched-2m/experiment.log 2>&1 < /dev/null &
echo $! > runs/matched-2m/experiment.pid
tail -F runs/matched-2m/experiment.log

# Recover one interrupted case without adding another full 2M budget
remaining_steps=$(uv run --extra rl python -c 'import json; from pathlib import Path; d=json.loads(Path("runs/matched-2m/hierarchical-seed-0/latest.json").read_text()); print(max(0, 2000000-d["checkpoint_steps"]))')
if [ "$remaining_steps" -gt 0 ]; then
  uv run --extra rl python -m src.game.rl train --resume runs/matched-2m/hierarchical-seed-0/latest.zip --steps "$remaining_steps" --n-envs 16 --eval-freq 65536 --eval-episodes 30 --eval-seed 10000 --output runs/matched-recovered
fi

# Three-seed combined-settings experiment; separate from the matched architecture control above
# Keep each checkpoint's .zip, .json, and .vecnormalize.pkl together
mkdir -p runs/mixed-v3-main-rebuild
nohup bash -c '
set -eu
for training_seed in 0 1 2; do
  output="runs/mixed-v3-main-rebuild/seed-${training_seed}"
  mkdir -p "$output"
  uv run --extra rl python -u -m src.game.rl train \
    --strategy hierarchical --steps 2000000 --seed "$training_seed" \
    --n-envs 8 --n-steps 512 --batch-size 256 --n-epochs 5 --device cpu \
    --learning-rate 0.0003 --lr-final 0.00005 --ent-coef 0.02 --ent-final 0.005 \
    --curriculum mixed --promotion-successes 30 --normalize \
    --max-steps 2400 --eval-freq 65536 --eval-episodes 30 --eval-seed 10000 \
    --output "$output" > "$output/training.log" 2>&1
  uv run --extra rl python -m src.game.rl evaluate --model "$output/best_model.zip" \
    --episodes 100 --seed 90000 --output "$output/evaluation.json" > "$output/evaluation.log" 2>&1
done
' > runs/mixed-v3-main-rebuild/experiment.log 2>&1 < /dev/null &
echo $! > runs/mixed-v3-main-rebuild/experiment.pid
tail -F runs/mixed-v3-main-rebuild/seed-0/training.log

# Automated checks
uv run --extra rl pytest
uv run --extra rl ruff check .
```
