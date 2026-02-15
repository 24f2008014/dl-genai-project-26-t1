import numpy as np
from pathlib import Path
import pandas as pd
import torch
import src.config as config
from src.dataset import TestDataset
from src.infer import make_predictions, ensemble_predictions
from src.models import ResNet, EffNetB0, ScratchCNN
from src.utils import get_train_stats

# Create the 3 model instances and load their weights
resnet_finetuned = ResNet(num_classes=10, freeze_backbone=False)
ckpt = torch.load(Path(config.CHECKPOINTS_DIR) / "resnet.ckpt", map_location="cpu", weights_only=False)
resnet_finetuned.load_state_dict(ckpt["model_state"])

effnet_finetuned = EffNetB0(num_classes=10, pretrained=False, freeze_backbone=True)
ckpt = torch.load(Path(config.CHECKPOINTS_DIR) / "effnetb0.ckpt", map_location="cpu", weights_only=False)
effnet_finetuned.load_state_dict(ckpt["model_state"])

scratch_cnn = ScratchCNN(num_classes=10)
ckpt = torch.load(Path(config.CHECKPOINTS_DIR) / "scratchcnn.ckpt", map_location="cpu", weights_only=False)
scratch_cnn.load_state_dict(ckpt["model_state"])

mean, std = get_train_stats(config.TRAIN_STATS_PATH) # Training set mel bin-wise mean and standard deviation
test_files = sorted([f for f in Path(config.TEST_DIR).iterdir() if f.is_file()])
test_set = TestDataset(test_files, mean=mean, std=std)

models = [effnet_finetuned, resnet_finetuned, scratch_cnn]
weights = [0.4, 0.4, 0.2]
preds = ensemble_predictions(models, test_set, weights=None, num_workers=4)
# preds = make_predictions(effnet_finetuned, test_set)

rows = []
for id, probs in preds.items():
    genre = config.idx_to_genre[np.argmax(probs)]
    rows.append((id, genre))
df = pd.DataFrame(rows, columns=["id", "genre"])
df["id"] = df["id"].str.extract(r"([0-9]+)") # Just retain the IDs
df.to_csv("submission.csv", index=False)