import pandas as pd

from tqdm import tqdm
from ast import literal_eval
from datasets import Dataset
from src.utils import make_answers_uniform

import prompts.prompt_templates as prompt_templates


class MyDataset:
    """
    A dataset processing class that prepares data for training and testing based on provided configurations.

    Args:
        cfg (dict): Configuration dictionary containing prompt name and uniform answer distribution flag.
    """

    def __init__(self, cfg):
        # Load the prompt template by name
        prompt_name = cfg["prompt_name"]
        self.PROMPT = getattr(prompt_templates, prompt_name)

        # Uniform answer distribution flag
        self.uniform_answer_distribution = cfg["uniform_answer_distribution"]

    def process(self, dataset_df, mode="train"):
        """
        Processes the input dataset and prepares it for training or testing.

        Args:
            dataset_df (pd.DataFrame): DataFrame containing the dataset to process.
            mode (str): Mode of operation ('train' or 'test').

        Returns:
            processed_dataset (datasets.Dataset): A processed Hugging Face Dataset object.
        """
        # Adjust answer distribution if required during training
        if mode == "train" and self.uniform_answer_distribution:
            dataset_df = make_answers_uniform(dataset_df)

        # Fill missing values with empty strings to prevent errors
        dataset_df.fillna("", inplace=True)

        records = []
        for _, row in dataset_df.iterrows():
            # Parse 'problems' column
            problems = literal_eval(row["problems"])

            # Parse 'reference' column; handle cases with missing references
            reference_data = row.get("reference", "")  # .get() for no RAG dataset
            reference = "\n".join([f"- {docs}" for idx, docs in enumerate(literal_eval(reference_data))])

            # Construct a record for each row
            record = {
                "id": "없음" if row["id"] == "" else row["id"],
                "paragraph": "없음" if row["paragraph"] == "" else row["paragraph"],
                "reference": "없음" if reference == "" else reference,
                "question": ("없음" if problems["question"] == "" else problems["question"]),
                "choices": "없음" if problems["choices"] == "" else problems["choices"],
                "answer": problems.get("answer", None),
                "question_plus": ("없음" if row["question_plus"] == "" else row["question_plus"]),
            }
            records.append(record)

        # Convert processed records to a DataFrame
        data_df = pd.DataFrame(records)

        processed = []
        for _, row in tqdm(data_df.iterrows(), desc="data", total=len(data_df)):
            # Create a string representation of the choices
            choices_str = "\n".join([f"{idx+1} - {choice}" for idx, choice in enumerate(row["choices"])])

            # Prepare system message from the prompt template
            system_msg = self.PROMPT["system_msg"]

            # Prepare user message using the prompt template
            user_msg = self.PROMPT["user_msg"].format(
                paragraph=row["paragraph"],
                reference=row["reference"],
                question=row["question"],
                question_plus=row["question_plus"],
                choices=choices_str,
            )

            # Add data based on the mode (train or test)
            if mode == "train":
                processed.append(
                    {
                        "id": row["id"],
                        "messages": [
                            {"role": "system", "content": system_msg},
                            {"role": "user", "content": user_msg},
                            {"role": "assistant", "content": f"{row['answer']}"},
                        ],
                        "label": row["answer"],
                    }
                )

            elif mode == "test":
                processed.append(
                    {
                        "id": row["id"],
                        "messages": [
                            {"role": "system", "content": system_msg},
                            {"role": "user", "content": user_msg},
                        ],
                        "label": row["answer"],
                        "len_choices": len(
                            row["choices"]
                        ),  # Include the number of choices for evaluation
                    }
                )

        # Convert processed data to Hugging Face Dataset
        processed_dataset = Dataset.from_pandas(pd.DataFrame(processed))
        return processed_dataset
