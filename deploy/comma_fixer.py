import torch
from params import ID2LABEL, LABEL2ID
from peft import PeftConfig, PeftModel
from transformers import AutoModelForTokenClassification, AutoTokenizer
import warnings
from logger import logger as base_logger
import time

logger = base_logger.bind(corr_id="CommaFixer ")


class CommaFixer:
    def __init__(self, config_path: str, device: str) -> None:
        self.config_path = config_path
        self.device = device
        logger.debug(f"Loading a model from {config_path} to {device}")
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

    def __infer(self, text):
        tokenized = self.tokenizer(text, return_tensors="pt", return_offsets_mapping=True, return_length=True)
        tokenized.to(self.model.device)
        start = time.time()
        with torch.inference_mode():
            logits = self.model(tokenized["input_ids"], tokenized["attention_mask"]).logits
        tokens = tokenized.tokens()
        predictions = torch.argmax(logits, dim=2).detach().cpu()
        labels = [self.model.config.id2label[prediction] for prediction in predictions[0].numpy()]
        logger.debug(f"Inference took {(time.time() - start):.3f} secs.")
        return tokens, labels, tokenized["offset_mapping"][0].detach().cpu().numpy()

    def __fix_commas_based_on_labels_and_offsets(
        self, labels: list[str], original_text: str, offset_map: list[tuple[int, int]]
    ) -> str:
        result = original_text
        commas_inserted = 0

        for i, label in enumerate(labels):
            current_offset = offset_map[i][1] + commas_inserted
            if label == "B-COMMA":
                result = result[:current_offset] + "," + result[current_offset:]
                commas_inserted += 1
        return result

    def remove_commas(self, text) -> str:
        text = text.replace(",", "")
        return text

    def fix_commas(self, text: str) -> str:
        text = self.remove_commas(text)
        _, predictions, offset = self.__infer(text)
        res = self.__fix_commas_based_on_labels_and_offsets(predictions, text, offset)
        return res


if __name__ == "__main__":
    warnings.warn("This module shouldn't be called directy.")
    comma_fixer = CommaFixer("just097/roberta-base-lora-comma-placement-r-8-alpha-32", "cpu")
    res = comma_fixer.fix_commas("One two three.")
    print(f"Formatted string with commas: {res}")
