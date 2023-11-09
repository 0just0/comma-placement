# Comma placement tool

Project goal is to create a usable tool to restore comma marks(`,`) in texts that do not contain them by mistake or on purpose.

This repo provides the training code for LoRa RoBerta model and some handy scripts to execute it from CLI or to pack in fully-functional microservice inside the Docker container.

## How to use it

### Install

* Pull the repository.
* Create a virtual env: ```python -m venv .venv```
* Install all nessesary depsendencies: ```pip install -r requirements.txt -r requirements-dev.txt```
* Now you are ready to use the tool.

### Use from CLI

```python comma_placement/inference.py --input <Your sentence without commas>```

### Create a web-server


## Idea

This problem can be approached as a token classification task. The idea is to train a transformer model on some text pairs using any relevant dataset from open-source(Wikitext?), remove commas from training examples and keep them for targets.

## Data

I am using [Wikitext](https://huggingface.co/datasets/wikitext) as a source of text data.
Final dataset used for training can be found here - [wiki-comma-placement](https://huggingface.co/datasets/just097/wiki-comma-placement). To reproduce pre-processing steps you may run ```python comma_placement/prepare_data.py``` but all the scripts are using the one from HF.

Data statistics:
* Train - 82.6k rows
* Validation - 20.7k rows
* Test - 19.7k rows

## Modeling

[Best model on HF](https://huggingface.co/just097/roberta-base-lora-comma-placement-r-16-alpha-32)

My experiments are mainly based on this [hf tutorial](https://huggingface.co/docs/peft/task_guides/token-classification-lora) and I am using LoRa to make training and iterations a little bit faster.

Models are saved to HF: [link](https://huggingface.co/just097).
Training logs and experiment tracking has been done via W&B: [link](https://wandb.ai/temnov-dmitry/wiki-comma-placement/overview)

To reproduce the best experiment you should run:

```python comma_placement/train.py```

There are a few params that might be helpful: `--use_wandb` and `--save_to_hf`. Set them to `False` if you don't want to track the experiment or push the model to hub. In this case, logging will be done to `stdout` and after the training the best model will be saved to `models/`.

All the nessesary configuration params are specified in ```comma_placement_config.py```. You might tweak them a little bit to change the dataset used to training or to use different params for LoRa or training process.

## Evaluation

To get the current `test` results you can run ```python comma_placement/evaluation.py```.

```--model``` param defines the model version that will be used at validation step(Set by default to the best one I managed to get).

As a baseline I am using https://huggingface.co/oliverguhr/fullstop-punctuation-multilang-large as a most popular one from HF. It is trained on a different data so performance on Wikitext is worth then provided in Model Card even though Wikitext is a pretty general text dataset without too complicated cases for commas.

| Model    | precision | recall | F1   |
|----------|-----------|--------|------|
| baseline*| 0.79     | 0.72   | 0.75  |
| My Model | 0.8378   | 0.8493 | 0.8435|
