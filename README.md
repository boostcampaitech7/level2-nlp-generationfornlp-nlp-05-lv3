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
python main.py --config {config_path} --mode {train/test}
```
- **`config`**: config.yaml의 경로(폴더 경로 말고 .yaml의 경로!)  
- **`mode`**: train, test 중 선택  

### config.yaml  
[여기](./config/sample.yaml)에서 확인, sample은 기존 baseline과 동일한 설정  
- **`data_path`**: train.csv, test.csv가 있는 폴더의 경로  
- **`prompt_path`**: 적용할 prompt의 yaml이 있는 파일 경로(폴더 경로 말고 .yaml의 경로!)  
- **`test_output_dir`**: `mode`가 `test`일 경우 추론 결과를 저장할 폴더 경로(해당 폴더가 사전에 존재해야 함)  
- **`model`**: 사용할 모델과 Training Arguments의 옵션들 지정(학습이 완료된 모델을 저장할 경로도 지정 가능)  
  - 사용할 모델에 따라 chat_template이 필요한 경우 True, 아닐 경우 False로 지정하기  
- **`peft`**: LoraConfig의 옵션들 지정  

### custom
- **프롬프트 수정**: [dataset.py](./src/dataset.py)에서 변경 가능  
  - `PROMPT_NO_QUESTION_PLUS`, `PROMPT_QUESTION_PLUS`, `process` 함수의 `messages`를 함께 수정해주기  
- **chat template**: [model.py](./src/model.py)의 line 31 참고  
- **추가적인 config**: [utils.py](./src/utils.py)에서 변수 추가 및 수정(config.yaml로도 가능하지만, 그 이상의 옵션들을 수정하고 싶을 경우)  