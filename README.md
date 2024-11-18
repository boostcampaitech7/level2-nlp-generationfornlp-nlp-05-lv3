## 베이스라인 코드 모듈화

### 폴더 구조
```bash
.
|-- README.md
|-- config
|   `-- sample.yaml
|-- outputs
|-- prompt
|   `-- base.yaml
|-- main.py
|-- outputs # test prediction outputs.csv
`-- src
    |-- dataset.py
    |-- model.py
    `-- utils.py
```

### 사용법
```bash
python main.py --config {config_path} --mode {train/valid/test}
```
- **`config`**: config.yaml의 경로(폴더 경로 말고 .yaml의 경로!)  
- **`mode`**: train, valid, test 중 선택  
  - train: train.csv로 모델 학습 → 학습이 완료된 모델로 valid.csv에 대한 추론을 이어서 수행  
  - valid: 학습이 완료된 모델(`test_name_or_path`)로 valid.csv에 대한 추론 수행  
  - test: 학습이 완료된 모델(`test_name_or_path`)로 test.csv에 대한 추론 수행  
- **inference 할 때 참고사항**  
  - valid.csv 추론 후 모델 저장 경로에 output_valid.csv 생성, `columns=[id, answer, pred]`(`answer`은 ground_truth)  
  - test.csv 추론 후 `test_output_dir`에 output_test.csv 생성, `columns=[id, answer]`  
  - output_valid.csv의 `pred`와 output_test.csv의 `answer`이 모델이 추론한 결과임  

### config.yaml  
[여기](./config/sample.yaml)에서 확인, sample은 기존 baseline과 동일한 설정  
- **`data_path`**: train.csv, valid.csv, test.csv가 있는 폴더의 경로  
- **`prompt_path`**: 적용할 prompt의 yaml이 있는 파일 경로(폴더 경로 말고 .yaml의 경로!)  
- **`test_output_dir`**: `mode`가 `test`일 경우 추론 결과를 저장할 폴더 경로(해당 폴더가 사전에 존재해야 함)  
- **`model`**: 사용할 모델과 Training Arguments의 옵션들 지정(학습이 완료된 모델을 저장할 경로도 지정 가능)  
  - `train_name_or_path`: 학습할 모델의 Hugging Face uid  
  - `test_name_or_path`: valid.csv, test.csv 추론 시 사용할 학습이 완료된 모델의 checkpoint 경로  
  - 사용할 모델에 따라 chat_template이 필요한 경우 True, 아닐 경우 False로 지정하기  
- **`peft`**: LoraConfig의 옵션들 지정  

### custom
- **프롬프트 수정**: [prompt.yaml](./prompt)에 만들고 추가
  - `no_question_plus_5`, `question_plus_5`, `no_question_plus_4`, `question_plus_4` 함께 지정하기
  - `config.yaml`의 `prompt_path` 지정하기
- **chat template**: [model.py](./src/model.py)의 line 94 참고  
- **추가적인 config**: [utils.py](./src/utils.py)에서 변수 추가 및 수정(config.yaml로도 가능하지만, 그 이상의 옵션들을 수정하고 싶을 경우)  
