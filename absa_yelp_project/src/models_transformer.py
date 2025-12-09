from __future__ import annotations
import os, json
import pandas as pd
from typing import Dict
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(axis=-1)
    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="binary", zero_division=0)
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}

def train_transformer_for_aspect(paths: Dict[str, str], models_dir: str, aspect: str, model_name: str, max_length: int, lr: float, batch_size: int, epochs: int, warmup_ratio: float, seed: int = 42, no_cuda: bool = False) -> Dict:
    os.makedirs(models_dir, exist_ok=True)
    train = pd.read_parquet(paths["train"])
    dev = pd.read_parquet(paths["dev"])

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tok(batch):
        return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=max_length)

    train_ds = Dataset.from_pandas(train[["text", "label"]])
    dev_ds = Dataset.from_pandas(dev[["text", "label"]])
    train_ds = train_ds.map(tok, batched=True)
    dev_ds = dev_ds.map(tok, batched=True)
    train_ds = train_ds.rename_column("label", "labels")
    dev_ds = dev_ds.rename_column("label", "labels")
    train_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    dev_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    args = TrainingArguments(
        output_dir=os.path.join(models_dir, f"hf_{aspect}"),
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=lr,
        num_train_epochs=epochs,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        warmup_ratio=warmup_ratio,
        seed=seed,
        logging_steps=50,
        report_to="none",
        no_cuda=no_cuda
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        compute_metrics=compute_metrics
    )

    trainer.train()
    metrics = trainer.evaluate()
    save_dir = os.path.join(models_dir, f"hf_{aspect}")
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    return {"aspect": aspect, "metrics_dev": metrics, "model_path": save_dir}
