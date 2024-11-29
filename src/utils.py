import os
import torch
import json
import ast
import random
import numpy as np
import pandas as pd

from collections import Counter
from glob import glob


def set_seed(random_seed):
    """
    Set the seed for random number generation to ensure reproducibility.

    Args:
        random_seed (int): Seed value to set for random generators.
    """
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)


def update_paths(config):
    """
    Update paths in the configuration file based on the experiment name.

    Args:
        config (dict): Configuration dictionary containing paths and experiment name.

    Returns:
        dict: Updated configuration dictionary with formatted paths.
    """
    experiment_name = config["model"]["experiment_name"]
    config["model"]["train"]["train_checkpoint_path"] = config["model"]["train"][
        "train_checkpoint_path"
    ].format(experiment_name=experiment_name)
    config["model"]["test"]["test_checkpoint_path"] = config["model"]["test"][
        "test_checkpoint_path"
    ].format(experiment_name=experiment_name)
    config["model"]["test"]["test_output_csv_path"] = config["model"]["test"][
        "test_output_csv_path"
    ].format(experiment_name=experiment_name)
    return config


def reset_token(experiment_name):
    """
    Reset specific token configurations in the tokenizer configuration files. This only works with Qwen2.5 model checkpoints.

    Args:
        experiment_name (str): Name of the experiment to locate the checkpoint directory.
    """
    base_path = f"checkpoints/{experiment_name}"
    checkpoints = glob(os.path.join(base_path, "checkpoint-*"))

    for checkpoint in checkpoints:
        tokenizer_config_path = os.path.join(checkpoint, "tokenizer_config.json")

        if not os.path.exists(tokenizer_config_path):
            continue

        with open(tokenizer_config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        # Remove the "chat_template" field if it exists
        if "chat_template" in config:
            del config["chat_template"]  # reset chat_template

        # Reset pad_token with the content of the "added_tokens_decoder"
        if "added_tokens_decoder" in config:
            pad_token_content = config["added_tokens_decoder"]["151665"]["content"]
            config["pad_token"] = pad_token_content  # reset pad_token

        with open(tokenizer_config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)


def make_answers_uniform(dataframe, seed=42):
    """
    Create a dataset with uniform answer distribution by shuffling and balancing the answer choices.

    Args:
        dataframe (pd.DataFrame): Input dataframe containing questions, choices, and answers.
        seed (int): Random seed for reproducibility. Default is 42.

    Returns:
        pd.DataFrame: Dataframe with uniform answer distribution.
    """
    random.seed(seed)

    # Randomly select one of the answers with the minimum count
    def get_random_min_answer(answer_counts):
        min_value = min(answer_counts.values())
        min_keys = [key for key, value in answer_counts.items() if value == min_value]
        return random.choice(min_keys)

    # Shuffle answer positions uniformly
    def shuffle_answer_position(row, answer_counts):
        choices = row["choices"]
        answer = row["answer"]

        # Get answer position with the minimum count
        answer_position = get_random_min_answer(answer_counts)
        answer_counts[answer_position] += 1

        # Swap the positions of answers
        new_choices = choices.copy()
        new_choices[answer_position - 1], new_choices[answer - 1] = (
            new_choices[answer - 1],
            new_choices[answer_position - 1],
        )

        return new_choices, answer_position

    # Convert row to string format for uniform dataset
    def dic_to_str(row):
        return str(
            {
                "question": row["question"],
                "choices": row["choices_u"],
                "answer": row["answer_u"],
            }
        )

    # Convert string to dictionary and extract question components
    dataframe["problems_dict"] = dataframe["problems"].apply(ast.literal_eval)
    dataframe["question"] = dataframe["problems_dict"].apply(lambda x: x["question"])
    dataframe["choices"] = dataframe["problems_dict"].apply(lambda x: x["choices"])
    dataframe["answer"] = dataframe["problems_dict"].apply(lambda x: x["answer"])

    # Split dataset by the number of choices (4-choice or 5-choice questions)
    sub_train_4 = dataframe[dataframe["choices"].apply(len) == 4].copy()
    sub_train_5 = dataframe[dataframe["choices"].apply(len) == 5].copy()

    # Initialize answer distributions
    answer_counts_4 = Counter({i: 0 for i in range(1, 5)})  # 4지 선다 정답 분포 초기화
    answer_counts_5 = Counter({i: 0 for i in range(1, 6)})  # 5지 선다 정답 분포 초기화

    # Shuffle answer choices for uniform distribution
    sub_train_4[["choices_u", "answer_u"]] = pd.DataFrame(
        sub_train_4.apply(
            shuffle_answer_position, axis=1, answer_counts=answer_counts_4
        ).tolist(),
        index=sub_train_4.index,
    )
    sub_train_5[["choices_u", "answer_u"]] = pd.DataFrame(
        sub_train_5.apply(
            shuffle_answer_position, axis=1, answer_counts=answer_counts_5
        ).tolist(),
        index=sub_train_5.index,
    )

    # Combine the two datasets
    train_uniform = (
        pd.concat([sub_train_4, sub_train_5])
        .sample(frac=1, random_state=42)
        .reset_index(drop=True)
    )

    # Convert dictionary to string for the uniform dataset
    train_uniform["problems_u"] = train_uniform.apply(dic_to_str, axis=1)
    train_uniform["problems"] = train_uniform["problems_u"]

    drop_columns = [
        "problems_dict",
        "question",
        "choices",
        "answer",
        "choices_u",
        "answer_u",
    ]
    train_uniform = train_uniform.drop(columns=drop_columns)

    return train_uniform


def print_answer_distribution(dataframe, tag="Train"):
    """
    Print the distribution of answers in the dataset.

    Args:
        dataframe (pd.DataFrame): Dataset to analyze.
        tag (str): Label for the dataset (e.g., 'Train', 'Test'). Default is 'Train'.
    """
    dataframe = dataframe.copy()
    dataframe["problems_dict"] = dataframe["problems"].apply(ast.literal_eval)
    dataframe["answer"] = dataframe["problems_dict"].apply(lambda x: x["answer"])

    print(f"{tag} Answer distribution")
    answer_counts = dataframe["answer"].value_counts().sort_index()
    answer_ratios = (
        dataframe["answer"].value_counts(normalize=True).sort_index().round(2)
    )
    result = pd.DataFrame({"Count": answer_counts, "Ratio": answer_ratios})

    print(result)
