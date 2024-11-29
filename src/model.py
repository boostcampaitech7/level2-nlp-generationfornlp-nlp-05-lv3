import torch
import evaluate
import numpy as np
import pandas as pd

from tqdm import tqdm
from trl import DataCollatorForCompletionOnlyLM
from unsloth import (
    FastLanguageModel,
    UnslothTrainer,
    UnslothTrainingArguments,
    is_bfloat16_supported,
)


def tokenize_dataset(dataset, tokenizer):
    """
    Tokenize the dataset using the provided tokenizer.

    Args:
        dataset (datasets.Dataset): Dataset to tokenize.
        tokenizer (transformers.PreTrainedTokenizer): Tokenizer for tokenizing the dataset.

    Returns:
        datasets.Dataset: Tokenized dataset.
    """
    tokenized_dataset = dataset.map(
        lambda element: tokenize_function(element, tokenizer),
        remove_columns=list(dataset.features),
        batched=True,
        num_proc=4,  # Number of processes to use for tokenization
        load_from_cache_file=True,
        desc="Tokenizing",
    )
    return tokenized_dataset


def tokenize_function(element, tokenizer):
    """
    Tokenize individual examples using the given tokenizer.

    Args:
        element (dict): Example from the dataset.
        tokenizer (transformers.PreTrainedTokenizer): Tokenizer for tokenizing examples.

    Returns:
        dict: Tokenized input_ids and attention_mask.
    """
    outputs = tokenizer(
        formatting_prompts_func(element, tokenizer),
        truncation=False,
        padding=False,
        return_overflowing_tokens=False,
        return_length=False,
    )
    return {
        "input_ids": outputs["input_ids"],
        "attention_mask": outputs["attention_mask"],
    }


def formatting_prompts_func(example, tokenizer):
    """
    Format prompts for tokenization using chat templates.

    Args:
        example (dict): Example containing messages to format.
        tokenizer (transformers.PreTrainedTokenizer): Tokenizer with chat template application.

    Returns:
        list: Formatted prompt strings for tokenization.
    """
    output_texts = []
    for message in example["messages"]:
        output_texts.append(
            tokenizer.apply_chat_template(
                message,
                tokenize=False,
            )
        )
    return output_texts


class MyModel:
    """
    A class to handle training and inference for a language model with Peft and Unsloth.

    Args:
        config (dict): Configuration dictionary for model and training setup.
        mode (str): Operating mode ('train' or 'test').
    """

    def __init__(self, config, mode):
        self.config = config
        self.model_c = config["model"]
        self.peft_c = config["peft"]
        self.unsloth_c = config["UnslothTrainingArguments"]

        # Load model and tokenizer
        if mode == "train":
            model_name = self.model_c["train"]["train_model_name"]
        elif mode == "test":
            model_name = self.model_c["test"]["test_checkpoint_path"]

        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=self.model_c["max_seq_length"],
            dtype=None,
            load_in_4bit=True,
        )

        # Apply Peft for training
        if mode == "train":
            self.model = FastLanguageModel.get_peft_model(
                self.model,
                r=self.peft_c["r"],
                lora_alpha=self.peft_c["lora_alpha"],
                lora_dropout=self.peft_c["lora_dropout"],
                target_modules=self.peft_c["target_modules"],
                bias=self.peft_c["bias"],
                use_gradient_checkpointing=self.peft_c["use_gradient_checkpointing"],
                random_state=self.config["seed"],
                use_rslora=self.peft_c["use_rslora"],
                loftq_config=None,
            )

        # Prepare model for inference in test mode
        elif mode == "test":
            self.model = FastLanguageModel.for_inference(self.model)

        # Define chat template
        self.tokenizer.chat_template = (
            "{% if messages[0]['role'] == 'system' %}"
            "{% set system_message = messages[0]['content'] %}"
            "{% endif %}"
            "{% if system_message is defined %}"
            "{{ system_message }}"
            "{% endif %}"
            "{% for message in messages %}"
            "{% set content = message['content'] %}"
            "{% if message['role'] == 'user' %}"
            "{{ '<start_of_turn>user\n' + content + '<end_of_turn>\n<start_of_turn>model\n' }}"
            "{% elif message['role'] == 'assistant' %}"
            "{{ content + '<end_of_turn>\n' }}"
            "{% endif %}"
            "{% endfor %}"
        )

        # Load accuracy metric
        self.acc_metric = evaluate.load("accuracy")

        # Map answer tokens to indices and vice versa
        self.int_output_map = {"1": 0, "2": 1, "3": 2, "4": 3, "5": 4}
        self.pred_choices_map = {0: "1", 1: "2", 2: "3", 3: "4", 4: "5"}

    def tokenize(self, processed_train):
        """
        Tokenize and split the processed training dataset.

        Args:
            processed_train (datasets.Dataset): Processed dataset to tokenize.
        """
        tokenized = tokenize_dataset(processed_train, self.tokenizer)
        tokenized = tokenized.filter(lambda x: len(x["input_ids"]) <= self.model_c["max_seq_length"])

        # Split into train and validation datasets if required
        if self.model_c["train_valid_split"]:
            tokenized = tokenized.train_test_split(test_size=0.1, seed=self.config["seed"])
            self.train_dataset = tokenized["train"]
            self.eval_dataset = tokenized["test"]
        else:
            self.train_dataset = tokenized
            self.eval_dataset = None

    def train(self, processed_train):
        """
        Train the model using the processed training dataset.

        Args:
            processed_train (datasets.Dataset): Dataset for training.
        """

        # Function to preprocess logits for metric computation
        def preprocess_logits_for_metrics(logits, labels):
            # Handle tuple format of logits
            logits = logits if not isinstance(logits, tuple) else logits[0]

            # Select logits corresponding to the answer tokens
            logit_idx = [
                self.tokenizer.vocab["1"],
                self.tokenizer.vocab["2"],
                self.tokenizer.vocab["3"],
                self.tokenizer.vocab["4"],
                self.tokenizer.vocab["5"],
            ]
            logits = logits[:, -2, logit_idx]  # -2: answer token, -1: eos token
            return logits

        # Function to compute evaluation metrics
        def compute_metrics(evaluation_result):
            logits, labels = evaluation_result

            # Replace padding labels with the pad token ID
            labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)

            # Decode tokenized labels into text
            labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

            # Extract answer tokens and map them to indices
            labels = list(map(lambda x: x.split("<end_of_turn>")[0].strip(), labels))
            labels = list(map(lambda x: self.int_output_map[x], labels))

            # Compute probabilities and predictions
            probs = torch.nn.functional.softmax(torch.tensor(logits), dim=-1)
            predictions = np.argmax(probs, axis=-1)

            # Calculate accuracy
            acc = self.acc_metric.compute(predictions=predictions, references=labels)
            return acc

        # Tokenize the processed training dataset
        self.tokenize(processed_train)

        # Prepare a data collator for the language model
        data_collator = DataCollatorForCompletionOnlyLM(
            response_template="<start_of_turn>model",
            tokenizer=self.tokenizer,
        )

        # Set tokenizer configurations for padding
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.tokenizer.padding_side = "right"

        # Check if evaluation is enabled during training
        do_eval = self.model_c["train_valid_split"]

        # Initialize the UnslothTrainer with the model, datasets, and training arguments
        trainer = UnslothTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset if do_eval else None,
            data_collator=data_collator,
            compute_metrics=compute_metrics if do_eval else None,
            preprocess_logits_for_metrics=(preprocess_logits_for_metrics if do_eval else None),
            dataset_num_proc=4,  # Use 4 processes for dataset operations
            # Training arguments
            args=UnslothTrainingArguments(
                do_train=True,
                do_eval=True if do_eval else False,
                per_device_train_batch_size=self.unsloth_c["per_device_train_batch_size"],
                per_device_eval_batch_size=self.unsloth_c["per_device_eval_batch_size"],
                gradient_accumulation_steps=self.unsloth_c["gradient_accumulation_steps"],
                warmup_ratio=self.unsloth_c["warmup_ratio"],
                num_train_epochs=self.unsloth_c["num_train_epochs"],
                learning_rate=float(self.unsloth_c["learning_rate"]),
                embedding_learning_rate=float(self.unsloth_c["embedding_learning_rate"]),
                fp16=not is_bfloat16_supported(),  # Use FP16 if BF16 is not supported
                bf16=is_bfloat16_supported(),  # Use BF16 if supported
                logging_steps=1,
                optim=self.unsloth_c["optim"],
                weight_decay=self.unsloth_c["weight_decay"],
                lr_scheduler_type=self.unsloth_c["lr_scheduler_type"],
                seed=self.config["seed"],
                max_seq_length=self.model_c["max_seq_length"],
                output_dir=self.model_c["train"]["train_checkpoint_path"],
                save_strategy=self.unsloth_c["save_strategy"],
                eval_strategy="epoch" if do_eval else "no",
                save_total_limit=self.unsloth_c["save_total_limit"],
                save_only_model=self.unsloth_c["save_only_model"],
                report_to="wandb",  # Reporting to WandB
            ),
        )

        # Start training
        trainer.train()

    def inference(self, processed_test, output_dir):
        """
        Perform inference using the processed test dataset and save results to a CSV file.

        Args:
            processed_test (datasets.Dataset): Dataset for inference.
            output_dir (str): Path to save the inference results.
        """
        infer_results = []

        # Set model to evaluation mode
        self.model.eval()

        # Disable gradient computation for inference
        with torch.inference_mode():
            for data in tqdm(processed_test):
                # Extract values for the data
                _id = data["id"]
                messages = data["messages"]
                len_choices = data["len_choices"]

                # Generate model outputs for the given input
                outputs = self.model(
                    self.tokenizer.apply_chat_template(
                        messages,
                        tokenize=True,
                        add_generation_prompt=True,
                        return_tensors="pt",
                    ).to("cuda")
                )

                # Extract logits for the last token & Collect logits for the target answer tokens
                logits = outputs.logits[:, -1].flatten().cpu()
                vocab = self.tokenizer.get_vocab()
                target_logit_list = [logits[vocab.get(str(i + 1))] for i in range(len_choices)]

                # Apply softmax to convert logits to probabilities
                probs = (
                    torch.nn.functional.softmax(torch.tensor(target_logit_list, dtype=torch.float32), dim=0)
                    .detach()
                    .cpu()
                    .numpy()
                )

                # Get the predicted answer based on the highest probability
                predict_value = self.pred_choices_map[np.argmax(probs, axis=-1)]

                infer_results.append({"id": _id, "answer": predict_value})

        # Save the inference results as a CSV file
        pd.DataFrame(infer_results).to_csv(output_dir, index=False)
