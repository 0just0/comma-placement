from datasets import load_dataset
from utils import data_process
from config import RAW_DATA

# Let's first load some raw text data from hugginface

dataset = load_dataset("wikitext", "wikitext-103-v1")

dataset_train = dataset["train"][0:100000]["text"]
samples_train = data_process.preprocess_texts(dataset_train)
data_process.save_to_txt(samples_train, f"{RAW_DATA}/raw_wiki_lines.txt")
