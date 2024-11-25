import torch
from trl import SFTConfig
from peft import LoraConfig
from transformers import BitsAndBytesConfig
import pandas as pd
import ast
import random
from collections import Counter
from unsloth import FastLanguageModel, UnslothTrainer, UnslothTrainingArguments, is_bfloat16_supported

# def get_peft_config(config):
#     config = LoraConfig(
#         r=config['r'],
#         lora_alpha=config['lora_alpha'],
#         lora_dropout=config['lora_dropout'],
#         target_modules=config['target_modules'],
#         bias=config['bias'],
#         task_type=config['task_type']
#     )

#     return config

def get_Unsloth_Training_Arguments(config, do_eval=False):
    UnslothTrainingArguments = UnslothTrainingArguments(
        do_train=True,
        do_eval=do_eval,
        lr_scheduler_type=config['sft']['lr_scheduler'],
        max_seq_length=config['model']['max_seq_length'],
        output_dir=config['model']['train']['train_checkpoint_path'],
        per_device_train_batch_size=int(config['sft']['batch_size']),
        per_device_eval_batch_size=int(config['sft']['batch_size']),
        num_train_epochs=int(config['sft']['epochs']),
        learning_rate=float(config['sft']['learning_rate']),
        embedding_learning_rate =float(config['sft']['embedding_learning_rate']),
        warmup_ratio=float(config['sft']['warmup_ratio']),
        optim=config['sft']['optim'],
        weight_decay=float(config['sft']['weight_decay']),
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps=1,
        save_strategy="epoch",
        eval_strategy="no",
        save_total_limit=3,
        save_only_model=True,
        report_to="wandb",
        gradient_accumulation_steps=config['sft']['gradient_accumulation_steps'],
    )
    if do_eval:
        UnslothTrainingArguments.eval_strategy = "epoch"
    else:
        UnslothTrainingArguments.eval_strategy = "no"

    return UnslothTrainingArguments

# def get_quant_config(config):
#     if config['compute_dtype'] == "float16":
#         dtype = torch.float16
#     elif config['compute_dtype'] == "float32":
#         dtype = torch.float32

#     if config['4bit_or_8bit'] == 4:
#         quantization_config = BitsAndBytesConfig(
#             load_in_4bit=True,
#             bnb_4bit_compute_dtype=dtype,
#             bnb_4bit_use_double_quant=config['double_quant'],
#             bnb_4bit_quant_type=config['quant_type']
#         )
#     elif config['4bit_or_8bit'] == 8:
#         quantization_config = BitsAndBytesConfig(
#             load_in_8bit=True,
#             bnb_8bit_compute_dtype=dtype,
#             bnb_8bit_use_double_quant=config['double_quant'],
#             bnb_8bit_quant_type=config['quant_type']
#         )

#     return quantization_config


def make_answers_uniform(dataframe, seed=42):
    """
    This function creates a dataset with uniform answer distribution.
    """
    random.seed(seed)

    # randomly select one of the answers with the minimum count
    def get_random_min_answer(answer_counts):
        min_value = min(answer_counts.values())
        min_keys = [key for key, value in answer_counts.items() if value == min_value]
        return random.choice(min_keys)

    # shuffle answer choices uniformly
    def shuffle_answer_position(row, answer_counts):
        choices = row["choices"]
        answer = row["answer"]

        # get answer position which has the minimum count
        answer_position = get_random_min_answer(answer_counts)
        answer_counts[answer_position] += 1

        # swap
        new_choices = choices.copy()
        new_choices[answer_position - 1], new_choices[answer - 1] = (
            new_choices[answer - 1],
            new_choices[answer_position - 1],
        )

        return new_choices, answer_position

    def dic_to_str(row):

        return str(
            {
                "question": row["question"],
                "choices": row["choices_u"],
                "answer": row["answer_u"],
            }
        )

    # convert string to dictionary and add columns
    dataframe["problems_dict"] = dataframe["problems"].apply(ast.literal_eval)
    dataframe["question"] = dataframe["problems_dict"].apply(lambda x: x["question"])
    dataframe["choices"] = dataframe["problems_dict"].apply(lambda x: x["choices"])
    dataframe["answer"] = dataframe["problems_dict"].apply(lambda x: x["answer"])

    # divide the dataset into two parts based on the number of choices
    sub_train_4 = dataframe[dataframe["choices"].apply(len) == 4].copy()
    sub_train_5 = dataframe[dataframe["choices"].apply(len) == 5].copy()

    # initialize answer distribution
    answer_counts_4 = Counter({i: 0 for i in range(1, 5)})  # 4지 선다 정답 분포 초기화
    answer_counts_5 = Counter({i: 0 for i in range(1, 6)})  # 5지 선다 정답 분포 초기화

    # shuffle answer choices uniformly
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

    # combine the two datasets
    train_uniform = (
        pd.concat([sub_train_4, sub_train_5])
        .sample(frac=1, random_state=42)
        .reset_index(drop=True)
    )

    # convert dictionary to string
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
    PRINT ANSWER DISTRIBUTION
    """
    dataframe = dataframe.copy()
    dataframe["problems_dict"] = dataframe["problems"].apply(ast.literal_eval)
    dataframe["answer"] = dataframe["problems_dict"].apply(lambda x: x["answer"])

    print(f"{tag} Answer distribution")
    answer_counts = dataframe["answer"].value_counts().sort_index()
    answer_ratios = dataframe["answer"].value_counts(normalize=True).sort_index().round(2)
    result = pd.DataFrame({"Count": answer_counts, "Ratio": answer_ratios})

    print(result)