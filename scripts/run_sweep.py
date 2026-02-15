import wandb
import src.config as config
from src.train import train_sweep_run

sweep_config = {
    "name": "ScratchCNN_sweep_full_set",
    "method": "random", # Random search
    "metric": {
        "goal": "maximize",
        "name": "val/macro_f1"
    },
    "parameters": {
        "lr": {
            "distribution": "log_uniform_values",
            "min": 1e-5,
            "max": 5e-4
        },
        "weight_decay": {
            "distribution": "log_uniform_values",
            "min": 1e-6,
            "max": 1e-3
        },
        "seed": { 
            "values": [1, 2, 3, 4, 5] # For seeding the RNG
        }
    }
}

sweep_id = wandb.sweep(sweep=sweep_config, project=config.WANDB_PROJECT_NAME)
n_runs = 10 # How many random configurations to try before stopping
print(f"Sweep ID: {sweep_id}")
wandb.agent(sweep_id, train_sweep_run(model_name="scratch", n_epochs=10), count=n_runs)