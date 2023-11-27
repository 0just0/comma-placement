# Comma placement tool

Project goal is to create a usable tool to restore comma marks(`,`) in texts that do not contain them by mistake or on purpose.

This repo provides the training code for LoRa RoBerta model and some handy scripts to execute it from CLI or to pack in fully-functional microservice inside the Docker container.

## How to use it

**Makefile**

I have created a simple Makefile to automate every step of the pipeline.
1. Create a venv - ```make virtualenv```
2. Activate it 😊
3. To install requirements - ```make requirements```
4. To add necessary directories - ```make dirs```
5. **Optional** If you want to use pre-commit -  ```make pre-commit-install```
6. To install spacy en_core_web_sm - ```make setup-data-prepare```
7. To reproduce a dataset creation - ```make run-data-prepare```
8. To reproduce training - ```make run-train```
9. To run evaluation on `test` subset - ```make run-eval```
10. To run a FastAPI server locally - ```make run-app```
11. To build an application Docker - ```make build-docker```. You may need to run with `sudo` on some systems.
12. To run an app in Docker - ```make run-docker```
13. To run tests - ```make run-tests```. They might fail if you don't have a running instance of a service.


## Manual steps with instructions:

### Install

* Pull the repository.
* Create a virtual env: ```python -m venv .venv```
* Install all necessary depsendencies: ```pip install -r requirements.txt -r requirements-dev.txt```
* Now you are ready to use the tool.

### Use from CLI

```python comma_placement/inference.py --input <Your sentence without commas>```

### Create a web-server

`deploy/` folder contains all the necessary components to start up a simple API server with comma_placement tool.

You can either run `./run.sh` to start a FastAPI service locally or you can build a Docker image.
**Attention!** I am using `pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime` to support GPU inference. If you want to execute only on CPU you should probably use some lighter and more optimized base image.

1. `docker build -t comma_fixer .` (It will be built to execute model on CPU by default. To change that - update `--device` param to `cuda:0`)
2. `docker run --name comma_fixer_0 -p 8008:80 -d comma_fixer` will boot up your instance on localhost:8008
3. Try the API at `docs/` page or make some requests from the terminal.
4. Example:
```
import requests
data = json.dumps({"input_text": "One Two three."})
response = requests.post("http://0.0.0.0:8008/", data)
response.json()["text_with_commas"]
>>> One, Two, three.
```

## Idea

This problem can be approached as a token classification task. The idea is to train a transformer model on some text:tags pairs using any relevant dataset from open-source(Wikitext?), remove commas from training examples and annotate the samples for tokens that should have a comma after them.

Example: ```"One two three." -> [0, 1, 0, 0]```

## Data

I am using [Wikitext](https://huggingface.co/datasets/wikitext) as a source of text data.

Final dataset used for training can be found here - [wiki-comma-placement](https://huggingface.co/datasets/just097/wiki-comma-placement).

To reproduce pre-processing steps you may run:
1. ```python -m spacy download en_core_web_sm```
2. ```python comma_placement/prepare_data.py```

All the scripts are using the preprocessed dataset from HF.

**Data statistics:**
| Subset   | Num of Rows |
|----------|-------------|
| Train    |    82.6k    |
| Validation |  20.7k    |
| Test     |    19.7k    |

## Modeling

[Best model on HF](https://huggingface.co/just097/roberta-base-lora-comma-placement-r-16-alpha-32)

My experiments are mainly based on this [hf tutorial](https://huggingface.co/docs/peft/task_guides/token-classification-lora) and I am using LoRa to make training and iterations a little bit faster.

Models are saved to HF: [link](https://huggingface.co/just097).
Training logs and experiment tracking has been done via W&B: [link](https://wandb.ai/temnov-dmitry/wiki-comma-placement/overview)

To reproduce the best experiment you should run:

```python comma_placement/train.py```

There are a few params that might be helpful: `--use_wandb` and `--save_to_hf`. Set them to `False` if you don't want to track the experiment or push the model to hub. In this case, logging will be done to `stdout` and after the training the best model will be saved to `models/`.

All the necessary configuration params are specified in ```comma_placement/config.py```. You might tweak them a little bit to change the dataset used to training or to use different params for LoRa or training process.

## Evaluation

To get the current `test` results you can run ```python comma_placement/evaluation.py```.

```--model``` param defines the model version that will be used at validation step(Set by default to the best one I managed to get).

As a baseline I am using https://huggingface.co/oliverguhr/fullstop-punctuation-multilang-large as a most popular one from HF. It is trained on a different data, so performance on Wikitext is worth then provided in Model Card, even though Wikitext is a pretty general text dataset without too complicated cases for commas.

Baseline evaluation can be reproduced via notebook ```notebooks/eval_baseline.ipynb```

| Model     | precision | recall | F1     |
|-----------|-----------|--------|--------|
| baseline* | 0.7262    | 0.6416 | 0.6813 |
| My Model  | 0.8378    | 0.8493 | 0.8435 |
