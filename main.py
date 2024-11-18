import os
import yaml
import torch
import random
import argparse
import numpy as np
import pandas as pd

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
    parser.add_argument("--config", "-c", type=str, default="config/sample.yaml")
    parser.add_argument("--mode", "-m", type=str, default="train")
    args = parser.parse_args()

    set_seed(42)

    with open(args.config) as f:
        config = yaml.full_load(f)

    dataset = MyDataset(config['prompt_path'])
    model = MyModel(config)

    if args.mode == "train":
        train_df = pd.read_csv(os.path.join(config['data_path'], 'train.csv'))
        valid_df = pd.read_csv(os.path.join(config['data_path'], 'valid.csv'))

        processed_train = dataset.process(train_df, "train")
        processed_valid = dataset.process(valid_df, "valid")
        model.train(processed_train, processed_valid)

        processed_valid = dataset.process(valid_df, mode="test")
        model.inference(processed_valid, "valid", output_dir=config['model']['output_dir'])
    elif args.mode == "valid":
        valid_df = pd.read_csv(os.path.join(config['data_path'], 'valid.csv'))
        processed_valid = dataset.process(valid_df, mode="test")
        model.inference(processed_valid, "valid", output_dir=config['model']['output_dir'])
    elif args.mode == "test":
        test_df = pd.read_csv(os.path.join(config['data_path'], 'test.csv'))
        processed_test = dataset.process(test_df, "test")
        model.inference(processed_test, "test", output_dir=config['test_output_dir'])