import os
import yaml
import argparse
import shutil
import pandas as pd

from src.model import MyModel
from src.dataset import MyDataset
from src.utils import set_seed, reset_token, update_paths


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", type=str, default="config/config.yaml")
    parser.add_argument("--mode", "-m", type=str, default="train")
    args = parser.parse_args()

    # Load YAML configuration file
    with open(args.config) as f:
        config = yaml.full_load(f)

    # Update paths based on the experiment name
    config = update_paths(config)

    # Set random seed for reproducibility
    set_seed(config["seed"])

    # Initialize dataset and model
    dataset = MyDataset(config["model"])
    model = MyModel(config, args.mode)

    base_path = "../contest_baseline_code"

    if args.mode == "train":
        # Training mode
        checkpoint_dir = config["model"]["train"]["train_checkpoint_path"]
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Save configuration file in the checkpoint directory
        shutil.copy(args.config, os.path.join(checkpoint_dir, "config.yaml"))

        # Process training data and train the model
        train_df = pd.read_csv(os.path.join(base_path, config["model"]["train"]["train_csv_path"]))
        processed_train = dataset.process(train_df, "train")
        model.train(processed_train)

        # Reset tokenizer token configurations
        reset_token(config["model"]["experiment_name"])

    elif args.mode == "test":
        # Testing mode
        test_df = pd.read_csv(os.path.join(base_path, config["model"]["test"]["test_csv_path"]))
        processed_test = dataset.process(test_df, "test")

        # Run inference and save results
        model.inference(processed_test, output_dir=os.path.join(base_path, config["model"]["test"]["test_output_csv_path"]),
        )
