import torch
from config import ID2LABEL, LABEL2ID
from peft import PeftConfig, PeftModel
from transformers import AutoModelForTokenClassification, AutoTokenizer

peft_model_id = "just097/roberta-base-lora-comma-placement-finetuned"
model = None
tokenizer = None


def infer(text):
    tokenized = tokenizer(
        text, return_tensors="pt", return_offsets_mapping=True, return_length=True
    )
    tokenized.to(model.device)
    with torch.inference_mode():
        logits = model(tokenized["input_ids"], tokenized["attention_mask"]).logits
    tokens = tokenized.tokens()
    predictions = torch.argmax(logits, dim=2).detach().cpu()
    labels = [
        model.config.id2label[prediction] for prediction in predictions[0].numpy()
    ]
    return tokens, labels, tokenized["offset_mapping"][0].detach().cpu().numpy()


def _should_insert_comma(label, result, current_offset) -> bool:
    # Only insert commas for the final token of a word, that is, if next word starts with a space.
    # TODO perhaps for low confidence tokens, we should use the original decision of the user in the input?
    return label == "B-COMMA"


def fix_commas_based_on_labels_and_offsets(
    labels: list[str], original_s: str, offset_map: list[tuple[int, int]]
) -> str:
    """
    This function returns the original string with only commas fixed, based on the predicted labels from the main
    model and the offsets from the tokenizer.
    :param labels: Predicted labels for the tokens.
    Should already be converted to string, since we will look for B-COMMA tags.
    :param original_s: The original string, used to preserve original spacing and punctuation.
    :param offset_map: List of offsets in the original string, we will only use the second integer of each pair
    indicating where the token ended originally in the string.
    :return: The string with commas fixed, and everything else intact.
    """
    result = original_s
    commas_inserted = 0

    for i, label in enumerate(labels):
        current_offset = offset_map[i][1] + commas_inserted
        if _should_insert_comma(label, result, current_offset):
            result = result[:current_offset] + "," + result[current_offset:]
            commas_inserted += 1
    return result


def convert_to_text(text: str) -> str:
    tokens, predictions, offset = infer(text)
    res = fix_commas_based_on_labels_and_offsets(predictions, text, offset)
    return res


if __name__ == "__main__":
    sample_sentences = [
        "one two three.",
        "Hey Mark how are you?",
        "This sentence shouldn't have any commas.",
        "You have to buy milk bread and coffee.",
        "This sentence shoud have comma here here and here however it doesn't.",
    ]
    config = PeftConfig.from_pretrained(peft_model_id)
    inference_model = AutoModelForTokenClassification.from_pretrained(
        config.base_model_name_or_path,
        num_labels=2,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )
    tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
    model = PeftModel.from_pretrained(inference_model, peft_model_id)
    model.cuda()
    model.eval()

    for i in sample_sentences:
        print(convert_to_text(i))
