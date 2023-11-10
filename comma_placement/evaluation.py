import argparse
from pprint import pprint

from .config import dataset_path, training_args
from datasets import load_dataset
from comma_fixer import CommaFixer
from metrics import compute_metrics
from transformers import DataCollatorForTokenClassification, Trainer

parser = argparse.ArgumentParser()
parser.add_argument(
    "--model",
    type=str,
    default="just097/roberta-base-lora-comma-placement-r-16-alpha-32",
    help="Please provide a model-id on HF",
)


model = None
tokenizer = None


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


if __name__ == "__main__":
    args = parser.parse_args()
    comma_fixer = CommaFixer(args.model, device="cpu")
    model, tokenizer = comma_fixer.model, comma_fixer.tokenizer
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

    print("TEST statistics:")
    pprint(trainer.predict(tokenized_wiki["test"])[2], indent=2)
