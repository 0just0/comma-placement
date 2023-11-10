import torch
from .config import ID2LABEL, LABEL2ID
from peft import PeftConfig, PeftModel
from transformers import AutoModelForTokenClassification, AutoTokenizer
import warnings
import spacy

nlp = spacy.load("en_core_web_sm")


class CommaFixer:
    def __init__(self, config_path: str, device: str) -> None:
        self.config_path = config_path
        self.device = device
        model, tokenizer = self.prepare_model(self.config_path, self.device)
        self.model = model
        self.tokenizer = tokenizer

    def prepare_model(self, config_path: str, device: str):
        config = PeftConfig.from_pretrained(config_path)
        inference_model = AutoModelForTokenClassification.from_pretrained(
            config.base_model_name_or_path,
            num_labels=2,
            id2label=ID2LABEL,
            label2id=LABEL2ID,
        )
        tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path, add_prefix_space=True)
        model = PeftModel.from_pretrained(inference_model, config_path)
        model = model.merge_and_unload()
        model.to(device)
        model.eval()
        return model, tokenizer

    def remove_commas(self, text) -> str:
        text = text.replace(",", "")
        return text

    def __split_by_sentence(self, text) -> list:
        doc = nlp(text)
        sentences = [str(sent) for sent in doc.sents]
        return sentences

    def __infer(self, text):
        tokenized = self.tokenizer(
            text, return_tensors="pt", padding=True, return_offsets_mapping=True, return_length=True
        )
        tokenized.to(self.model.device)
        with torch.inference_mode():
            logits = self.model(tokenized["input_ids"], tokenized["attention_mask"]).logits
        tokens = tokenized.tokens()
        predictions = torch.argmax(logits, dim=2).detach().cpu()
        labels = [self.model.config.id2label[prediction] for prediction in predictions[0].numpy()]
        return tokens, labels, tokenized["offset_mapping"][0].detach().cpu().numpy()

    def __fix_commas_based_on_labels_and_offsets(
        self, labels: list[str], original_text: str, offset_map: list[tuple[int, int]]
    ) -> str:
        result = original_text
        commas_inserted = 0

        for i, label in enumerate(labels):
            current_offset = offset_map[i][1] + commas_inserted
            if label == "B-COMMA" and result[current_offset].isspace():
                result = result[:current_offset] + "," + result[current_offset:]
                commas_inserted += 1
        return result

    def fix_commas(self, text: str) -> str:
        result = []
        text = self.remove_commas(text)
        if len(text) > 512:
            text = self.__split_by_sentence(text)
        else:
            text = [text]
        for t in text:
            _, predictions, offset = self.__infer(t)
            res = self.__fix_commas_based_on_labels_and_offsets(predictions, t, offset)
            result.append(res)

        return " ".join(result)


if __name__ == "__main__":
    warnings.warn("This module shouldn't be called directy.")
