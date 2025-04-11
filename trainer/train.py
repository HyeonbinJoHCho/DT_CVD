import os
import re
from transformers import TrainingArguments, Trainer

def get_latest_checkpoint(checkpoint_dir: str) -> str:
    """
    Find the latest checkpoint directory in a given folder.

    Args:
        checkpoint_dir (str): Path to the directory containing checkpoint folders.

    Returns:
        str: Path to the latest checkpoint directory, or None if none found.
    """
    pattern = re.compile(r"^checkpoint-\d+$")
    candidates = [
        os.path.join(checkpoint_dir, d)
        for d in os.listdir(checkpoint_dir)
        if pattern.match(d) and os.path.isdir(os.path.join(checkpoint_dir, d))
    ]
    return max(candidates, key=os.path.getmtime) if candidates else None

def train_model(model, tokenized_train, tokenized_val, data_collator, model_path: str = None):
    """
    Set up Trainer and perform training.

    Args:
        model: The model to train.
        tokenized_train: Tokenized training dataset.
        tokenized_val: Tokenized validation dataset.
        data_collator: Custom data collator for MLM masking.
        model_path (str): Path where model checkpoints and logs will be saved.

    Returns:
        Trainer: The trainer instance (already trained).
    """
    training_args = TrainingArguments(
        output_dir=model_path,
        evaluation_strategy="epoch",
        num_train_epochs=300,
        per_device_train_batch_size=2048,
        per_device_eval_batch_size=2048,
        learning_rate=1e-3,
        weight_decay=0.00001,
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        push_to_hub=False,
        bf16=True,
        dataloader_num_workers=16,
    )

    latest_ckpt = get_latest_checkpoint(model_path)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        data_collator=data_collator,
    )

    trainer.train(resume_from_checkpoint=latest_ckpt)
    return trainer
