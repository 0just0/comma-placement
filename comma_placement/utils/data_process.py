import re

space_remover = r"\s([,?.!:;](?:\s|$))"
quote_space_remover = r'"\s*([^"]*?)\s*"'
tokenizer = None


def remove_spaces(samples):
    samples = [sample.strip() for sample in samples]
    samples = [re.sub(space_remover, r"\1", sample) for sample in samples]
    samples = [re.sub(quote_space_remover, r'"\1"', sample) for sample in samples]
    samples = [sample.replace(" '", "'") for sample in samples]
    return samples


def remove_empty(samples: list) -> list:
    samples = [sample for sample in samples if sample]
    return samples


def remove_titles(samples: list) -> list:
    samples = [sample for sample in samples if "=" not in sample]
    return samples


def remove_short(samples: list) -> list:
    samples = [sample for sample in samples if len(sample) > 300]
    return samples


def remove_unk(samples: list) -> list:
    samples = [sample for sample in samples if "<unk>" not in sample]
    return samples


def preprocess_texts(samples: list) -> list:
    samples = remove_titles(samples)
    samples = remove_spaces(samples)
    samples = remove_empty(samples)
    samples = remove_short(samples)
    samples = remove_unk(samples)
    return samples


def save_to_file(samples: list, path: str):
    with open(f"{path}", "w") as f:
        f.writelines([sample + "\n" for sample in samples])


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
