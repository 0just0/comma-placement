.PHONY: clean dirs virtualenv lint requirements push pull reproduce

#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
PYTHON_INTERPRETER = python3

#################################################################################
# COMMANDS                                                                      #
#################################################################################

## Create virtualenv.
## Activate with the command:
## source env/bin/activate
virtualenv:
	$(PYTHON_INTERPRETER) -m venv .venv
	$(info "Activate with the command 'source env/bin/activate'")

## Install Python Dependencies.
## Make sure you activate the virtualenv first!
requirements:
	$(PYTHON_INTERPRETER) -m pip install -U pip setuptools wheel
	$(PYTHON_INTERPRETER) -m pip install -r requirements.txt -r requirements-dev.txt

## Create directories that are ignored by git but required for the project
dirs:
	mkdir -p data/raw data/processed models checkpoints

## Delete all compiled Python files
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

## To use the pre-commit hooks
pre-commit-install:
	pre-commit install

setup-data-prepare:
	python -m spacy download en_core_web_sm

run-data-prepare:
	cd comma_placement; python prepare_data.py

run-train:
	cd comma_placement; python train.py

run-eval:
	cd comma_placement; python evaluation.py

start-app:
	cd deploy; ./run.sh

build-docker:
	cd deploy; docker build -t comma_fixer .

run-docker:
	docker run --name comma_fixer_0 -p 8008:80 -d comma_fixer

#################################################################################
# PROJECT RULES                                                                 #
#################################################################################



#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := reproduce

# Inspired by <http://marmelab.com/blog/2016/02/29/auto-documented-makefile.html>
# sed script explained:
# /^##/:
# 	* save line in hold space
# 	* purge line
# 	* Loop:
# 		* append newline + line to hold space
# 		* go to next line
# 		* if line starts with doc comment, strip comment character off and loop
# 	* remove target prerequisites
# 	* append hold space (+ newline) to line
# 	* replace newline plus comments by `---`
# 	* print line
# Separate expressions are necessary because labels cannot be delimited by
# semicolon; see <http://stackoverflow.com/a/11799865/1968>
.PHONY: help
help:
	@echo "$$(tput bold)Available rules:$$(tput sgr0)"
	@echo
	@sed -n -e "/^## / Missing" $Missing \
	| LC_ALL='C' sort --ignore-case \
	| awk -F '---' \
		-v ncol=$$(tput cols) \
		-v indent=19 \
		-v col_on="$$(tput setaf 6)" \
		-v col_off="$$(tput sgr0)" \
	'Missing \
			printf "%s ", words[i]; \
		} \
		printf "\n"; \
	}' \
	| more $(shell test $(shell uname) = Darwin && echo '--no-init --raw-control-chars')
