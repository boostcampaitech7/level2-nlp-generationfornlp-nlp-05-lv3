import yaml
import pandas as pd
import copy
from tqdm import tqdm
from ast import literal_eval
from datasets import Dataset
from src.utils import make_answers_uniform, print_answer_distribution
import prompts.prompt_templates as prompt_templates
import numpy as np

class MyDataset:
    def __init__(self, cfg):
        prompt_name = cfg['prompt_name']
        self.is_rag = cfg['rag']
        self.RAG_PROMPT = getattr(prompt_templates, prompt_name)
        self.NO_RAG_PROMPT = copy.deepcopy(self.RAG_PROMPT)
        for key, template in self.NO_RAG_PROMPT['user_msg'].items():
            self.NO_RAG_PROMPT['user_msg'][key] = template.replace(
                "참고:\n{reference}\n\n", ""
            )

        self.uniform_answer_distribution = cfg['uniform_answer_distribution']

    def process(self, dataset_df, mode='train'):
        # uniform answer distribution
        if mode == "train" and self.uniform_answer_distribution:
            # print answer distribution
            print()
            print("*" * 50)
            print_answer_distribution(dataset_df, "Original")
            print("*" * 50)                        
            dataset_df = make_answers_uniform(dataset_df)
            print_answer_distribution(dataset_df, "Processed Train")
            print("*" * 50)        
        
        records = []
        for _, row in dataset_df.iterrows():
            problems = literal_eval(row['problems'])
            
            # RAG reference 처리
            reference = ""
            if self.is_rag:
                reference_data = row.get('reference', "")  # reference가 없을 수도 있으니 .get 사용
                reference = "\n".join([f"- {docs}" for idx, docs in enumerate(literal_eval(reference_data))])
                                    
            # record 딕셔너리 생성
            record = {
                'id': row['id'],
                'paragraph': row['paragraph'],
                'reference': reference,
                'question': problems['question'],
                'choices': problems['choices'],
                'answer': problems.get('answer', None),
                "question_plus": problems.get('question_plus', None),
            }
            if 'question_plus' in problems:
                record['question_plus'] = problems['question_plus']
            records.append(record)
                
        data_df = pd.DataFrame(records)

        processed = []
        for _, row in tqdm(data_df.iterrows(), desc='data', total=len(data_df)):
            choices_str = "\n".join([f"{idx+1} - {choice}" for idx, choice in enumerate(row["choices"])])
            
            # system_msg
            system_msg = self.RAG_PROMPT['system_msg']      

            # user_msg with RAG
            if pd.notna(row['paragraph']) and isinstance(row['paragraph'], str) and len(row['paragraph']) < 500 and row['reference'] != "" and self.is_rag:
                # 보기(question_plus)가 있을 때
                if row["question_plus"]:
                    user_msg = self.RAG_PROMPT['user_msg']['question_plus_5'].format(
                        paragraph=row["paragraph"],
                        reference=row['reference'],
                        question=row["question"],
                        question_plus=row["question_plus"],
                        choices=choices_str
                    )
                # 보기(question_plus)가 없을 때 
                else:
                    user_msg = self.RAG_PROMPT['user_msg']['no_question_plus_5'].format(
                        paragraph=row["paragraph"],
                        reference=row['reference'],
                        question=row["question"],
                        choices=choices_str
                    )
                    
            # user_msg without RAG
            else:
                # 보기(question_plus)가 있을 때
                if row["question_plus"]:
                    user_msg = self.NO_RAG_PROMPT['user_msg']['question_plus_5'].format(
                        paragraph=row["paragraph"],
                        question=row["question"],
                        question_plus=row["question_plus"],
                        choices=choices_str
                    )
                # 보기(question_plus)가 없을 때 
                else:
                    user_msg = self.NO_RAG_PROMPT['user_msg']['no_question_plus_5'].format(
                        paragraph=row["paragraph"],
                        question=row["question"],
                        choices=choices_str
                    )
            
            if mode == "train":
                processed.append({
                    "id": row["id"],
                    "messages": [
                        {"role": "system", "content": system_msg},
                        {"role": "user", "content": user_msg},
                        {"role": "assistant", "content": f"{row['answer']}"}
                    ],
                    "label": row["answer"]
                })
            elif mode == "test":
                processed.append({
                    "id": row["id"],
                    "messages": [
                        {"role": "system", "content": system_msg},
                        {"role": "user", "content": user_msg},
                    ],
                    "label": row["answer"],
                    "len_choices": len(row["choices"])
                })         
            
        if mode == "train":
            processed_dataset = Dataset.from_pandas(pd.DataFrame(processed))
            return processed_dataset
        elif mode == "test":
            return processed
    