import os
import yaml
import torch
import random
import argparse
import numpy as np
import pandas as pd
import shutil

from src.model import MyModel
from src.dataset import MyDataset

def set_seed(random_seed):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", type=str, default="config/config.yaml")
    parser.add_argument("--mode", "-m", type=str, default="train")
    args = parser.parse_args()

    set_seed(42)

    with open(args.config) as f:
        config = yaml.full_load(f)

    dataset = MyDataset(config['model'])
    model = MyModel(config, args.mode)
    base_path = "../contest_baseline_code"

    if args.mode == "train":
        checkpoint_dir = config['model']['train']['train_checkpoint_path']
        shutil.copy(args.config, os.path.join(checkpoint_dir, "config.yaml"))
        train_df = pd.read_csv(os.path.join(base_path, config['model']['train']['train_csv_path']))
        processed_train = dataset.process(train_df, "train")
        model.train(processed_train)

    elif args.mode == "test":
        test_df = pd.read_csv(os.path.join(base_path, config['model']['test']['test_csv_path']))
        processed_test = dataset.process(test_df, "test")
        model.inference(processed_test, output_dir=os.path.join(base_path, config['model']['test']['test_output_csv_path']))