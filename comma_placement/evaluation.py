import evaluate
import numpy as np
import torch
from config import ID2LABEL, LABEL2ID, LABEL_LIST, dataset_path, training_args
from datasets import load_dataset
from inference import prepare_model
from metrics import compute_metrics
from transformers import DataCollatorForTokenClassification, Trainer
from pprint import pprint


def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)

    labels = []
    for i, label in enumerate(examples[f"tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs


model, tokenizer = prepare_model("just097/roberta-base-lora-comma-placement-r-8-alpha-32", device="cpu")

wiki_comma_placement = load_dataset(dataset_path)
tokenized_wiki = wiki_comma_placement.map(tokenize_and_align_labels, batched=True)
data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_wiki["train"],
    eval_dataset=tokenized_wiki["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

pprint(trainer.evaluate(tokenized_wiki["test"]), indent=2)
