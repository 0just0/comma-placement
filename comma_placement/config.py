# Some constants and params for data processing, training and evaluating

RAW_DATA = "../data/raw"
PROCESSED_DATA = "../data/processed"

DATASET_NAME = "wiki-comma-placement"
DATASET_PATH = f"{PROCESSED_DATA}/{DATASET_NAME}"


ID2LABEL = {0: "O", 1: "B-COMMA"}
LABEL2ID = {"O": 0, "B-COMMA": 1}
