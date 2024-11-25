import yaml
import pandas as pd
from tqdm import tqdm
from ast import literal_eval
from datasets import Dataset
from src.utils import make_answers_uniform, print_answer_distribution
import prompts.prompt_templates as prompt_templates

class MyDataset:
    def __init__(self, cfg):
        prompt_name = cfg['prompt_name']
        self.prompt = getattr(prompt_templates, prompt_name)
        self.uniform_answer_distribution = cfg['uniform_answer_distribution']

    def process(self, dataset_df, mode='train'):
        if mode == "train":
            if self.uniform_answer_distribution:
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
            record = {
                'id': row['id'],
                'paragraph': row['paragraph'],
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
            system_msg = self.prompt['system_msg']      

            # debug
            # 보기(question_plus)가 있을 때
            if row["question_plus"]:
                user_msg = self.prompt['user_msg']['question_plus_5'].format(
                    paragraph=row["paragraph"],
                    question=row["question"],
                    question_plus=row["question_plus"],
                    choices=choices_str
                )
            # 보기(question_plus)가 없을 때 
            else:
                user_msg = self.prompt['user_msg']['no_question_plus_5'].format(
                    paragraph=row["paragraph"],
                    question=row["question"],
                    choices=choices_str
                )

            # if len(row["choices"]) == 5:
            #     # 보기(question_plus)가 있을 때
            #     if row["question_plus"]:
            #         user_msg = self.prompt['user_msg']['question_plus_5'].format(
            #             paragraph=row["paragraph"],
            #             question=row["question"],
            #             question_plus=row["question_plus"],
            #             choices=choices_str
            #         )
            #     # 보기(question_plus)가 없을 때 
            #     else:
            #         user_msg = self.prompt['user_msg']['no_question_plus_5'].format(
            #             paragraph=row["paragraph"],
            #             question=row["question"],
            #             choices=choices_str
            #         )
            # elif len(row["choices"]) == 4:
            #     if row["question_plus"]:
            #         user_msg = self.prompt['user_msg']['question_plus_4'].format(
            #             paragraph=row["paragraph"],
            #             question=row["question"],
            #             question_plus=row["question_plus"],
            #             choices=choices_str
            #         )
            #     # 보기(question_plus)가 없을 때 
            #     else:
            #         user_msg = self.prompt['user_msg']['no_question_plus_4'].format(
            #             paragraph=row["paragraph"],
            #             question=row["question"],
            #             choices=choices_str
            #         )
            
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

        print('*'*50)                
        print('Log - Prompt 확인:')
        data_sample = processed[0]
        print('<System>')
        print(data_sample["messages"][0]["content"])
        print('<User>')
        print(data_sample["messages"][1]["content"])
        print('*'*50)        
            
        if mode == "train":
            processed_dataset = Dataset.from_pandas(pd.DataFrame(processed))
            return processed_dataset
        elif mode == "test":
            return processed
        
        
class MyDatasetRag:
    def __init__(self, cfg):
        prompt_name = cfg['prompt_name']
        self.prompt_base = getattr(prompt_templates, "BASE_PROMPT")
        self.prompt = getattr(prompt_templates, prompt_name)
        self.uniform_answer_distribution = cfg['uniform_answer_distribution']

    def process(self, dataset_df, mode='train'):
        if mode == "train":
            if self.uniform_answer_distribution:
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
            reference = "\n".join([f"- {docs}" for idx, docs in enumerate(literal_eval(row['reference']))])
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
            system_msg = self.prompt['system_msg']      

            # debug
            if len(row['paragraph']) < 500:
                # 보기(question_plus)가 있을 때
                if row["question_plus"]:
                    user_msg = self.prompt['user_msg']['question_plus_5'].format(
                        paragraph=row["paragraph"],
                        reference=row['reference'],
                        question=row["question"],
                        question_plus=row["question_plus"],
                        choices=choices_str
                    )
                # 보기(question_plus)가 없을 때 
                else:
                    user_msg = self.prompt['user_msg']['no_question_plus_5'].format(
                        paragraph=row["paragraph"],
                        reference=row['reference'],
                        question=row["question"],
                        choices=choices_str
                    )
            else:
                # 보기(question_plus)가 있을 때
                if row["question_plus"]:
                    user_msg = self.prompt_base['user_msg']['question_plus_5'].format(
                        paragraph=row["paragraph"],
                        question=row["question"],
                        question_plus=row["question_plus"],
                        choices=choices_str
                    )
                # 보기(question_plus)가 없을 때 
                else:
                    user_msg = self.prompt_base['user_msg']['no_question_plus_5'].format(
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

        print('*'*50)                
        print('Log - Prompt 확인1:')
        data_sample = processed[0]
        print('<System>')
        print(data_sample["messages"][0]["content"])
        print('<User>')
        print(data_sample["messages"][1]["content"])
        print('*'*50)     
                     
        print('Log - Prompt 확인2:')
        data_sample = processed[-1]
        print('<System>')
        print(data_sample["messages"][0]["content"])
        print('<User>')
        print(data_sample["messages"][1]["content"])
        print('*'*50)    
            
        if mode == "train":
            processed_dataset = Dataset.from_pandas(pd.DataFrame(processed))
            return processed_dataset
        elif mode == "test":
            return processed