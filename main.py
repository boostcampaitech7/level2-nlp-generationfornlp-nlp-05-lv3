import os
import yaml
import torch
import random
import argparse
import json
import numpy as np
import pandas as pd
import shutil

from glob import glob
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
    
def reset_token(experiment_name):
    base_path = f"checkpoints/{experiment_name}"
    checkpoints = glob(os.path.join(base_path, "checkpoint-*"))

    for checkpoint in checkpoints:
        tokenizer_config_path = os.path.join(checkpoint, "tokenizer_config.json")
        
        if not os.path.exists(tokenizer_config_path):
            continue

        with open(tokenizer_config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        
        if "chat_template" in config:
            del config["chat_template"]  # chat_template 삭제
        if "added_tokens_decoder" in config:
            pad_token_content = config["added_tokens_decoder"]["128004"]["content"]
            config["pad_token"] = pad_token_content  # pad_token 변경

        with open(tokenizer_config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", type=str, default="config/config.yaml")
    parser.add_argument("--mode", "-m", type=str, default="train")
    args = parser.parse_args()


    with open(args.config) as f:
        config = yaml.full_load(f)

    # yaml파일 앵커 파싱
    experiment_name = config['model']['experiment_name']
    config['model']['train']['train_checkpoint_path'] = config['model']['train']['train_checkpoint_path'].format(experiment_name=experiment_name)
    config['model']['test']['test_checkpoint_path'] = config['model']['test']['test_checkpoint_path'].format(experiment_name=experiment_name)
    config['model']['test']['test_output_csv_path'] = config['model']['test']['test_output_csv_path'].format(experiment_name=experiment_name)

    set_seed(config['seed'])

    dataset = MyDataset(config['model'])
    model = MyModel(config, args.mode)
    base_path = "../contest_baseline_code"

    if args.mode == "train":
        checkpoint_dir = config['model']['train']['train_checkpoint_path']
        os.makedirs(checkpoint_dir, exist_ok=True)
        shutil.copy(args.config, os.path.join(checkpoint_dir, "config.yaml"))
        train_df = pd.read_csv(os.path.join(base_path, config['model']['train']['train_csv_path']))
        processed_train = dataset.process(train_df, "train")
        model.train(processed_train)
        # inference 전에 chat_template을 삭제하고 pad_token을 리셋
        reset_token(experiment_name)

    elif args.mode == "test":
        test_df = pd.read_csv(os.path.join(base_path, config['model']['test']['test_csv_path']))
        processed_test = dataset.process(test_df, "test")
        model.inference(processed_test, output_dir=os.path.join(base_path, config['model']['test']['test_output_csv_path']))