from trl import SFTConfig
from peft import LoraConfig

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
        load_best_model_at_end=True,
    )

    return sft_config
