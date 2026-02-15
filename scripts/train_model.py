import time
import torch
import wandb
import src.config as config
from src.train import train_run

GROUP_NAME = "scratch-full-top-2"
run_configs = [
    {"lr": 0.000089791, "weight_decay": 0.0000044606, "seed": 4},
    {"lr": 0.00039839, "weight_decay": 0.00031892, "seed": 4}
]

for i, cfg in enumerate(run_configs, start=1):
    run_name = f"scratch-full-{i}"
    run = wandb.init(project=config.WANDB_PROJECT_NAME, config=cfg, group=GROUP_NAME, name=run_name)
    print(f"STARTING RUN {i}/{len(run_configs)}: {run_name}, config={cfg}")
    try:
        train_run(model_name="scratch", i=i, cfg=cfg, n_epochs=30)
    except Exception as e: # Just try the next run if one fails
        print(f"RUN {i} FAILED: {e}")
    finally:
        run.finish()
        torch.cuda.empty_cache() # Clear GPU memory in between runs
        time.sleep(5)