import yaml
import pandas as pd
from tqdm import tqdm
from ast import literal_eval
from datasets import Dataset
from src.utils import make_answers_uniform, print_answer_distribution


class MyDataset:
    def __init__(self, prompt_path):
        with open(prompt_path) as f:
            self.prompt = yaml.full_load(f)

    def process(self, dataset_df, mode='train'):
        if mode == "train":
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

            if len(row["choices"]) == 5:
                # 보기(question_plus)가 있을 때
                if row["question_plus"]:
                    user_msg = self.prompt['question_plus_5'].format(
                        paragraph=row["paragraph"],
                        question=row["question"],
                        question_plus=row["question_plus"],
                        choices=choices_str
                    )
                # 보기(question_plus)가 없을 때 
                else:
                    user_msg = self.prompt['no_question_plus_5'].format(
                        paragraph=row["paragraph"],
                        question=row["question"],
                        choices=choices_str
                    )
            elif len(row["choices"]) == 4:
                if row["question_plus"]:
                    user_msg = self.prompt['question_plus_4'].format(
                        paragraph=row["paragraph"],
                        question=row["question"],
                        question_plus=row["question_plus"],
                        choices=choices_str
                    )
                # 보기(question_plus)가 없을 때 
                else:
                    user_msg = self.prompt['no_question_plus_4'].format(
                        paragraph=row["paragraph"],
                        question=row["question"],
                        choices=choices_str
                    )
            
            # PROMPT 시스템 메시지 수정
            system_message = {
                "role": "system",
                "content": (
                    "시험 문제를 푸는 똑똑한 학생으로서 다음 문제의 답을 구하세요.\n"
                    f"지문을 읽고, 질문에 대한 답을 1부터 {len(row['choices'])}까지의 선택지 중에 한 개만 골라서 대답해야 합니다."
                )
            }

            if mode == "train":
                processed.append({
                    "id": row["id"],
                    "messages": [
                        system_message,
                        {"role": "user", "content": user_msg},
                        {"role": "assistant", "content": f"{row['answer']}"}
                    ],
                    "label": row["answer"]
                })
            elif mode == "test":
                processed.append({
                    "id": row["id"],
                    "messages": [
                        system_message,
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