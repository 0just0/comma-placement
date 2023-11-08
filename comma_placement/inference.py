import argparse

from comma_fixer import CommaFixer

parser = argparse.ArgumentParser()
parser.add_argument(
    "--model",
    type=str,
    default="just097/roberta-base-lora-comma-placement-finetuned",
    help="Please provide a model-id on HF",
)
parser.add_argument(
    "--input", type=str, default="One two three.", help="Enter text without commas."
)
parser.add_argument("--device", default="cpu")
args = parser.parse_args()

peft_model_id = args.model
device = args.device


if __name__ == "__main__":
    sample_sentence = args.input
    comma_fixer = CommaFixer(peft_model_id, device)
    res = comma_fixer.fix_commas(sample_sentence)
    print(f"Formatted string with commas:\n {res}")
