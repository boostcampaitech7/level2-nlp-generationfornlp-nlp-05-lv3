![GIFMaker_me (1)](https://github.com/user-attachments/assets/d7a01735-9089-4abd-9d17-17154a5a872c)

![GIFMaker_me](https://github.com/user-attachments/assets/e35bf18d-9b2f-4b40-bfd5-ee3cbe9693ef)


https://github.com/user-attachments/assets/4448f058-6571-4037-9fb9-dfd8f86d5291


## 베이스라인 코드 모듈화

### 폴더 구조
```bash
level2-nlp-generationfornlp-nlp-05-lv3/  
├── .git/  
├── .github/  
├── checkpoints/                                # 학습된 모델 체크포인트 저장 폴더  
│   ├── (experiment_name)/              # 실험 이름  
│   │   ├── checkpoint-1111             # 모델 체크포인트  
│   │   └── checkpoint-2222  
│   └── .gitkeep  
├── config/  
│   └── config.yaml                           # 모든 설정 관리 파일  
├── notebooks/  
│   └── eda.ipynb  
├── prompt/  
│   ├── AI_provocation_prompt.yaml # 프롬프트 저장 파일(AI 자극 프롬프트)  
│   ├── base.yaml                                # 프롬프트 저장 파일(베이스라인 코드 프롬프트)  
├── src/  
│   ├── dataset.py  
│   ├── model.py  
│   ├── utils.py  
│   └── wandb/  
├── .gitignore  
├── main.py  
└── README.md  
```

### 사용법
```bash
python main.py --config {config_path} --mode {train/test}
```
- **`config`**: config.yaml의 경로(폴더 경로 말고 .yaml의 경로!)  
- **`mode`**: train, test 중 선택  

### config.yaml  
[여기](./config/config.yaml)에서 확인
- **`train_model_name`**: 학습할 모델 이름(Hugging Face)
- **`train_csv_path`**: train csv 파일 경로
- **`train_checkpoint_path`**: 학습한 체크포인트 저장 경로 

- **`test_checkpoint_path`**: 추론할 체크포인트 경로  
- **`test_csv_path`**: test csv 파일 경로
- **`test_output_csv_path`**: 리더보드 제출용 csv 파일 경로

- **`prompt_path`**: 프롬프트 파일 경로
- **`uniform_answer_distribution`**: True: 정답 분포 균등화, False: train 데이터의 정답 분포 그대로 사용
- **`train_valid_split`**: True: train 0.9, valid 0.1 스플릿, False: valid로 나누지 않고 train만 사용

- **`max_seq_length`**: 입력 토큰 최대 길이

- **`sft`**: train.csv, test.csv가 있는 폴더의 경로  
- **`peft`**: train.csv, test.csv가 있는 폴더의 경로  
