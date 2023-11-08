import torch
from config import ID2LABEL, LABEL2ID
from peft import PeftConfig, PeftModel
from transformers import AutoModelForTokenClassification, AutoTokenizer

import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--model",
    type=str,
    default="just097/roberta-base-lora-comma-placement-finetuned",
    help="Please provide a model-id on HF",
)
parser.add_argument("--input", type=str, default="One two three.", help="Enter text without commas.")
parser.add_argument("--device", default="cpu")
args = parser.parse_args()

peft_model_id = args.model
model = None
tokenizer = None
device = args.device


def prepare_model(config_path: str, device: str):
    config = PeftConfig.from_pretrained(peft_model_id)
    inference_model = AutoModelForTokenClassification.from_pretrained(
        config.base_model_name_or_path,
        num_labels=2,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )
    tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path, add_prefix_space=True)
    model = PeftModel.from_pretrained(inference_model, peft_model_id)
    model.to(device)
    model.eval()
    return model, tokenizer


def infer(text):
    tokenized = tokenizer(text, return_tensors="pt", return_offsets_mapping=True, return_length=True)
    tokenized.to(model.device)
    with torch.inference_mode():
        logits = model(tokenized["input_ids"], tokenized["attention_mask"]).logits
    tokens = tokenized.tokens()
    predictions = torch.argmax(logits, dim=2).detach().cpu()
    labels = [model.config.id2label[prediction] for prediction in predictions[0].numpy()]
    return tokens, labels, tokenized["offset_mapping"][0].detach().cpu().numpy()


def fix_commas_based_on_labels_and_offsets(
    labels: list[str], original_text: str, offset_map: list[tuple[int, int]]
) -> str:
    result = original_text
    commas_inserted = 0

    for i, label in enumerate(labels):
        current_offset = offset_map[i][1] + commas_inserted
        if label == "B-COMMA":
            result = result[:current_offset] + "," + result[current_offset:]
            commas_inserted += 1
    return result


def convert_to_text(text: str) -> str:
    _, predictions, offset = infer(text)
    res = fix_commas_based_on_labels_and_offsets(predictions, text, offset)
    return res


if __name__ == "__main__":
    sample_sentence = args.input
    model, tokenizer = prepare_model(peft_model_id, args.device)
    print(convert_to_text(sample_sentence))
