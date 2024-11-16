import torch
from trl import SFTConfig
from peft import LoraConfig
from transformers import BitsAndBytesConfig

def get_peft_config(config):
    config = LoraConfig(
        r=config['r'],
        lora_alpha=config['lora_alpha'],
        lora_dropout=config['lora_dropout'],
        target_modules=config['target_modules'],
        bias=config['bias'],
        task_type=config['task_type']
    )

    return config

def get_sft_config(config):
    sft_config = SFTConfig(
        do_train=True,
        do_eval=True,
        lr_scheduler_type=config['lr_scheduler'],
        max_seq_length=1024,
        output_dir=config['output_dir'],
        per_device_train_batch_size=int(config['batch_size']),
        per_device_eval_batch_size=int(config['batch_size']),
        num_train_epochs=int(config['epochs']),
        learning_rate=float(config['learning_rate']),
        weight_decay=float(config['weight_decay']),
        logging_steps=500,
        save_strategy="epoch",
        eval_strategy="epoch",
        save_total_limit=2,
        save_only_model=True,
        report_to="wandb",
    )

    return sft_config

def get_quant_config(config):
    if config['compute_dtype'] == "float16":
        dtype = torch.float16
    elif config['compute_dtype'] == "float32":
        dtype = torch.float32

    if config['4bit_or_8bit'] == 4:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=config['double_quant'],
            bnb_4bit_quant_type=config['quant_type']
        )
    elif config['4bit_or_8bit'] == 8:
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True
        )

    return quantization_config
