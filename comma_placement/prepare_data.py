import json

import spacy
from config import PROCESSED_DATA, RAW_DATA, DATASET_PATH, DATASET_NAME
from datasets import load_dataset, load_from_disk
from sklearn.model_selection import train_test_split
from utils import data_process

# Let's first load some raw text data from hugginface
nlp = spacy.load("en_core_web_sm")
dataset = load_dataset("wikitext", "wikitext-103-v1")


def sentence_to_sample(sentence: str) -> str:
    sentence = sentence.strip()
    words = [word.text for word in nlp(sentence)]
    tags = []
    clean_words = []
    for i in range(len(words) - 1):
        if words[i] == ",":
            continue
        if words[i + 1] == ",":
            clean_words.append(words[i])
            tags.append(1)
        else:
            clean_words.append(words[i])
            tags.append(0)
    clean_words.append(words[-1])
    tags.append(0)
    assert len(tags) == len(clean_words)
    return json.dumps({"tokens": clean_words, "tags": tags})


dataset_train = dataset["train"][0:100000]["text"]
samples_train = data_process.preprocess_texts(dataset_train)
txt_path = f"{RAW_DATA}/raw_wiki_lines.txt"
data_process.save_to_file(samples_train, txt_path)

with open(txt_path, "r") as f:
    data = f.readlines()

formatted_samples = []

for i in data:
    sentences = i.split(".")
    for j in sentences:
        sentence = j.strip()
        if len(sentence.split()) < 10:
            continue
        formatted_samples.append(sentence_to_sample(j + "."))
json_path = f"{RAW_DATA}/wiki_data.json"
data_process.save_to_file(formatted_samples, json_path)


with open(json_path, "r") as f:
    data = f.readlines()

train_lines, test_lines = train_test_split(data, test_size=0.16, random_state=42)
train_lines, val_lines = train_test_split(train_lines, test_size=0.2, random_state=42)

processed_paths = [
    f"{PROCESSED_DATA}/wiki_data_{step}.json" for step in ["train, validation, test"]
]

for step, lines in zip(processed_paths, [train_lines, val_lines, test_lines]):
    with open(f"{step}", "w") as f:
        f.writelines(lines)


dataset_structure = {
    "train": processed_paths[0],
    "validation": processed_paths[1],
    "test": processed_paths[2],
}

processed_dataset = load_dataset("json", data_files=dataset_structure)
processed_dataset.save_to_disk(DATASET_PATH)

# Upload to hub if needed.
dd = load_from_disk(DATASET_PATH)
dd.push_to_hub(DATASET_NAME)
