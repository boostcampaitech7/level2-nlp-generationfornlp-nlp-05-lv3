import os
import torch
import evaluate
import numpy as np
import pandas as pd
from tqdm import tqdm

from peft import AutoPeftModelForCausalLM
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils import get_peft_config, get_sft_config, get_quant_config


class MyModel():
    def __init__(self, config):
        self.config = config
        self.peft_c = config['peft']
        self.model_c = config['model']

        # metric 로드
        self.acc_metric = evaluate.load("accuracy")

        # 정답 토큰 매핑
        self.int_output_map = {"1": 0, "2": 1, "3": 2, "4": 3, "5": 4}
        self.pred_choices_map = {0: "1", 1: "2", 2: "3", 3: "4", 4: "5"}
    
    def tokenize(self, processed_train):
        tokenizer = self.tokenizer

        def tokenize_fn(element):
            output_texts = []
            for i in range(len(element["messages"])):
                output_texts.append(
                    tokenizer.apply_chat_template(
                        element["messages"][i],
                        tokenize=False,
                    )
                )

            outputs = tokenizer(
                output_texts,
                truncation=False,
                padding=False,
                return_overflowing_tokens=False,
                return_length=False
            )

            return {
                "input_ids": outputs["input_ids"],
                "attention_mask": outputs["attention_mask"],
            }

        tokenized = processed_train.map(
            tokenize_fn,
            remove_columns=list(processed_train.features),
            batched=True,
            num_proc=4,
            load_from_cache_file=True,
            desc="Tokenizing"
        )

        # vram memory 제약으로 인해 인풋 데이터의 길이가 1024 초과인 데이터는 제외하였습니다.
        # *힌트: 1024보다 길이가 더 긴 데이터를 포함하면 더 높은 점수를 달성할 수 있을 것 같습니다!
        tokenized = tokenized.filter(lambda x: len(x["input_ids"]) <= 1024)
        # validation 데이터셋 고정할 수도 있을 것
        tokenized = tokenized.train_test_split(test_size=0.1, seed=42)

        self.train_dataset = tokenized["train"]
        self.eval_dataset = tokenized["test"]

    def train(self, processed_train):
        quant_config = get_quant_config(self.config['quantization'])
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_c['name_or_path'],
            torch_dtype=torch.float16,
            trust_remote_code=True,
            quantization_config=quant_config,
            device_map="auto"
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_c['name_or_path'],
            trust_remote_code=True,
        )

        if self.model_c["chat_template"]:
            self.tokenizer.chat_template = "{% if messages[0]['role'] == 'system' %}{% set system_message = messages[0]['content'] %}{% endif %}{% if system_message is defined %}{{ system_message }}{% endif %}{% for message in messages %}{% set content = message['content'] %}{% if message['role'] == 'user' %}{{ '<start_of_turn>user\n' + content + '<end_of_turn>\n<start_of_turn>model\n' }}{% elif message['role'] == 'assistant' %}{{ content + '<end_of_turn>\n' }}{% endif %}{% endfor %}"
        
        # pad token 설정
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.tokenizer.padding_side = 'right'

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

        peft_config = get_peft_config(self.peft_c)
        sft_config = get_sft_config(self.model_c)

        trainer = SFTTrainer(
            model=self.model,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            data_collator=data_collator,
            tokenizer=self.tokenizer,
            compute_metrics=compute_metrics,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
            peft_config=peft_config,
            args=sft_config
        )

        trainer.train()
    
    def inference(self, processed_test, output_dir):
        self.model = AutoPeftModelForCausalLM.from_pretrained(
            self.model_c["name_or_path"],
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_c["name_or_path"],
            trust_remote_code=True,
        )

        infer_results = []

        self.model.to("cuda")
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
        
        pd.DataFrame(infer_results).to_csv(os.path.join(output_dir, "output.csv"), index=False)