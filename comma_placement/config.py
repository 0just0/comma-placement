# Some constants and params for data processing, training and evaluating
from transformers import TrainingArguments
from peft import LoraConfig, TaskType

RAW_DATA = "../data/raw"
PROCESSED_DATA = "../data/processed"

DATASET_NAME = "wiki-comma-placement"
DATASET_PATH = f"{PROCESSED_DATA}/{DATASET_NAME}"


ID2LABEL = {0: "O", 1: "B-COMMA"}
LABEL2ID = {"O": 0, "B-COMMA": 1}
LABEL_LIST = ["O", "B-COMMA"]

base_model = "roberta-base"
lr = 1e-3
batch_size = 32
num_epochs = 5

# Lora
r = 8
alpha = 32

dataset_path = f"just097/{DATASET_NAME}"  # My formatted dataset
model_name = f"roberta-base-lora-comma-placement-r-{r}-alpha-{alpha}"
checkpoints_path = f"../models/{model_name}"

training_args = TrainingArguments(
    checkpoints_path,
    learning_rate=lr,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=num_epochs,
    weight_decay=0.01,
    fp16=True,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    report_to="wandb",
    run_name=model_name,
    logging_steps=1,
    metric_for_best_model="f1",
)

peft_config = LoraConfig(
    task_type=TaskType.TOKEN_CLS,
    inference_mode=False,
    r=r,
    lora_alpha=alpha,
    lora_dropout=0.1,
    modules_to_save=["classifier"],
)
