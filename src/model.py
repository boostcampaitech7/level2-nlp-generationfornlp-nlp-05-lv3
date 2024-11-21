import os
import torch
import evaluate
import numpy as np
import pandas as pd
from tqdm import tqdm

from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM

from src.utils import get_sft_config
from unsloth import FastLanguageModel
from peft import AutoPeftModelForCausalLM, LoraConfig


def formatting_prompts_func(example, tokenizer):
    output_texts = []
    for message in example["messages"]:
        output_texts.append(
            tokenizer.apply_chat_template(
                message,
                tokenize=False,
            )
        )
    return output_texts

def tokenize_function(element, tokenizer):
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

def tokenize_dataset(dataset, tokenizer):
    tokenized_dataset = dataset.map(
        lambda element: tokenize_function(element, tokenizer),  # tokenizer 전달
        remove_columns=list(dataset.features),
        batched=True,
        num_proc=4,  # 병렬 처리
        load_from_cache_file=True,
        desc="Tokenizing",
    )
    return tokenized_dataset



class MyModel():
    def __init__(self, config, mode):
        self.config = config
        self.peft_c = config['peft']
        self.model_c = config['model']

        if mode == "train":
            self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                model_name=config['model']['train']['train_model_name'],
                max_seq_length=config['model']['max_seq_length'],
                dtype=getattr(torch, config['model']['torch_dtype']),
                load_in_4bit=config['model']['load_in_4bit']
            )
            
            self.model = FastLanguageModel.get_peft_model(
                self.model,
                r=self.peft_c['r'],
                lora_alpha=self.peft_c['lora_alpha'],
                lora_dropout=self.peft_c['lora_dropout'],
                target_modules=self.peft_c['target_modules'],
                bias=self.peft_c['bias'],
                use_gradient_checkpointing=self.peft_c['use_gradient_checkpointing'],
                random_state=self.peft_c['random_state'],
                use_rslora=self.peft_c['use_rslora'],
                loftq_config=self.peft_c['loftq_config'],
            )

        elif mode == "test":
            self.model = AutoPeftModelForCausalLM.from_pretrained(
                self.model_c['test']['test_checkpoint_path'],
                trust_remote_code=True,
                device_map="auto",
            )            
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_c['test']['test_checkpoint_path'],
                trust_remote_code=True,
            )
        
        # gemma-ko-2b에 chat template 직접 지정
        if self.model_c["chat_template"]:
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

        # metric 로드
        self.acc_metric = evaluate.load("accuracy")

        # 정답 토큰 매핑
        self.int_output_map = {"1": 0, "2": 1, "3": 2, "4": 3, "5": 4}
        self.pred_choices_map = {0: "1", 1: "2", 2: "3", 3: "4", 4: "5"}
    
    def tokenize(self, processed_train):
        tokenized = tokenize_dataset(processed_train, self.tokenizer)

        tokenized = tokenized.filter(lambda x: len(x["input_ids"]) <= self.model_c['max_seq_length'])
        
        if self.model_c['train_valid_split']:
            tokenized = tokenized.train_test_split(test_size=0.1, seed=42)
            print("*"*50)
            print("Log: Train Vaild 9:1로 나눴음")
            print("*"*50)
            self.train_dataset = tokenized["train"]
            self.eval_dataset = tokenized["test"]
        else:
            print("*"*50)
            print("Log: Train Vaild 안나눴음")
            print("*"*50)
            self.train_dataset = tokenized
            self.eval_dataset = None

    def train(self, processed_train):
        self.tokenize(processed_train)

        response_template = "<start_of_turn>model"
        data_collator = DataCollatorForCompletionOnlyLM(
            response_template=response_template,
            tokenizer=self.tokenizer,
        )

        # 모델의 logits 를 조정하여 정답 토큰 부분만 출력하도록 설정
        def preprocess_logits_for_metrics(logits, labels):
            logits = logits if not isinstance(logits, tuple) else logits[0]
            logit_idx = [self.tokenizer.vocab["1"], self.tokenizer.vocab["2"], self.tokenizer.vocab["3"], self.tokenizer.vocab["4"], self.tokenizer.vocab["5"]]
            logits = logits[:, -2, logit_idx] # -2: answer token, -1: eos token
            return logits
        
        # metric 계산 함수
        def compute_metrics(evaluation_result):
            logits, labels = evaluation_result

            # 토큰화된 레이블 디코딩
            labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
            labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
            labels = list(map(lambda x: x.split("<end_of_turn>")[0].strip(), labels))
            labels = list(map(lambda x: self.int_output_map[x], labels))

            probs = torch.nn.functional.softmax(torch.tensor(logits), dim=-1)
            predictions = np.argmax(probs, axis=-1)

            acc = self.acc_metric.compute(predictions=predictions, references=labels)
            return acc

        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.tokenizer.padding_side = 'right'
        
        do_eval = self.eval_dataset is not None
        sft_config = get_sft_config(self.config, do_eval=do_eval)

        trainer_args = {
            "model": self.model,
            "train_dataset": self.train_dataset,
            "data_collator": data_collator,
            "tokenizer": self.tokenizer,
            "args": sft_config
        }
        
        if do_eval:
            trainer_args["eval_dataset"] = self.eval_dataset
            trainer_args["compute_metrics"] = compute_metrics
            trainer_args["preprocess_logits_for_metrics"] = preprocess_logits_for_metrics

        trainer = SFTTrainer(**trainer_args)

        trainer.train()
    
    def inference(self, processed_test, output_dir):
        infer_results = []

        self.model.eval()
        with torch.inference_mode():
            for data in tqdm(processed_test):
                _id = data["id"]
                messages = data["messages"]
                len_choices = data["len_choices"]

                outputs = self.model(
                    self.tokenizer.apply_chat_template(
                        messages,
                        tokenize=True,
                        add_generation_prompt=True,
                        return_tensors="pt"
                    ).to("cuda")
                )

                logits = outputs.logits[:, -1].flatten().cpu()

                target_logit_list = [logits[self.tokenizer.vocab[str(i + 1)]] for i in range(len_choices)]

                probs = (
                    torch.nn.functional.softmax(
                        torch.tensor(target_logit_list, dtype=torch.float32)
                    ).detach().cpu().numpy()
                )

                predict_value = self.pred_choices_map[np.argmax(probs, axis=-1)]
                infer_results.append({"id": _id, "answer": predict_value})
        
        pd.DataFrame(infer_results).to_csv(output_dir, index=False)