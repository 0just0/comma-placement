import os
import wandb
from config import ID2LABEL, LABEL2ID, training_args, base_model, dataset_path, model_name, peft_config
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
)
from metrics import compute_metrics

os.environ["WANDB_PROJECT"] = "wiki-comma-placement"  # name your W&B project
os.environ["WANDB_LOG_MODEL"] = "checkpoint"  # log all model checkpoints


training_args = training_args

print(training_args)

tokenizer = AutoTokenizer.from_pretrained(base_model, add_prefix_space=True)


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


# Prepare a dataset for training. ###
wiki_comma_placement = load_dataset(dataset_path)
tokenized_wiki = wiki_comma_placement.map(tokenize_and_align_labels, batched=True)
data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

### Set up models for training ###
model = AutoModelForTokenClassification.from_pretrained(base_model, num_labels=2, id2label=ID2LABEL, label2id=LABEL2ID)

model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_wiki["train"],
    eval_dataset=tokenized_wiki["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()
trainer.evaluate(tokenized_wiki["test"], metric_key_prefix="test")

model.push_to_hub(model_name)
