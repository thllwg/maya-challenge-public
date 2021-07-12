.PHONY: clean data lint requirements sync_models_to_s3 sync_models_from_s3

#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
BUCKET = maya-models
PROFILE = default
PROJECT_NAME = maya-challenge
PYTHON_INTERPRETER = python3
MAYA_DATA_URL = https://drive.google.com/uc?id=10ECLfyKC32DLHqO7mS3p01Au9yBrW4r6
S3_URL= https://radosgw.public.os.wwu.de

ifeq (,$(shell which conda))
HAS_CONDA=False
else
HAS_CONDA=True
endif

#################################################################################
# COMMANDS                                                                      #
#################################################################################

## Install Python Dependencies
requirements: test_environment
	$(PYTHON_INTERPRETER) -m pip install -U pip setuptools wheel
	$(PYTHON_INTERPRETER) -m pip install -r requirements.txt

## Make Dataset
data: 
	@echo ">>> Downloading data from Google Drive."
	mkdir -p data/raw/
	gdown -O data/raw/data.zip $(MAYA_DATA_URL)
	@echo ">>> Unzipping."
	unzip -q data/raw/data.zip -d data/raw && rm data/raw/data.zip
	mv data/raw/S1\ and\ S2\ TIFF\ file\ structure.pdf references/S1\ and\ S2\ TIFF\ file\ structure.pdf
	$(PYTHON_INTERPRETER) src/data/make_dataset.py data/raw data/interim data/processed --val_percent 0.00

## Downloads a preprocessed dataset that can be directly used for training 
processed_data: 
	@echo ">>> Downloading preprocessed dataset from ERDA."
	mkdir -p data/processed/
	wget -P data/processed/ https://sid.erda.dk/share_redirect/HIwSfMceop
	@echo ">>> Unzipping."
	unzip -q data/processed/HIwSfMceop -d data/processed
	@echo ">>> Cleaning up."
	mv data/processed/processed/* data/processed && rm data/processed/HIwSfMceop


Model = ""
Type = "unet"
predict:
	rm -rf predictions/masks_train/*
	$(PYTHON_INTERPRETER) src/models/predict_model.py -m $(Model) -mt $(Type) -i data/processed -o predictions/masks_train --test-augmentation

predict_ensemble:
	rm -rf data/processed/ensemble_predictions/*
	$(PYTHON_INTERPRETER) src/models/ensemble_predictions.py  -i data/processed -o data/processed/ensemble_predictions/ --test-augmentation True @ensemble.txt

## Produces a submission-ready zip file
submission_file:
	rm -f submission.zip
	$(PYTHON_INTERPRETER) src/competition/make_submission.py predictions/masks_train submission/ --clean-masks --replace-aguadas

submission_from_ensemble: predict_ensemble
	rm -f submission.zip
	$(PYTHON_INTERPRETER) src/competition/make_submission.py data/processed/ensemble_predictions/ensemble submission/ --clean-masks --replace-aguadas


## Delete all compiled Python files and remove data set
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete
	find data/raw/ -type f -name "*.tif*" -delete
	find data/raw/ -type d -empty -delete
	find data/interim/ -type f -name "*.npy" -delete
	find data/interim/ -type d -empty -delete
	find data/processed/ -type f -name "*.npy" -delete
	find data/processed/ -type d -empty -delete


## Lint using flake8
lint:
	flake8 src

## Upload Data to S3
sync_models_to_s3:
ifeq (default,$(PROFILE))
	aws --endpoint-url $(S3_URL) s3 sync runs/stefan_29/ s3://$(BUCKET)/data/stefan_29/ --exclude "*0.pth" --exclude "logs.txt" --exclude "BestModel_*.pth"
else
	aws --endpoint-url $(S3_URL) s3 sync runs/ s3://$(BUCKET)/data/ --profile $(PROFILE) --exclude "*0.pth" --exclude "logs.txt" --exclude "BestModel_*.pth"
endif

## Download Data from S3
sync_models_from_s3:
ifeq (default,$(PROFILE))
	aws --endpoint-url $(S3_URL) s3 sync s3://$(BUCKET)/data/ runs/ --exclude "*0.pth" --exclude "logs.txt" --exclude "BestModel_*.pth"
else
	aws --endpoint-url $(S3_URL) s3 sync s3://$(BUCKET)/data/ runs/ --profile $(PROFILE) --exclude "*0.pth" --exclude "logs.txt" --exclude "BestModel_*.pth"
endif

## Set up python interpreter environment
create_environment:
ifeq (True,$(HAS_CONDA))
		@echo ">>> Detected conda, creating conda environment."
ifeq (3,$(findstring 3,$(PYTHON_INTERPRETER)))
	conda env create -f /tmp/conda-tmp/environment.yml
else
	conda create --name $(PROJECT_NAME) python=2.7
endif
	@echo ">>> New conda env created. Activate with:\nsource activate $(PROJECT_NAME)"
else
	@echo ">>> WARNING: Could not detect conda environment. The use of alternative interpreter environments is discouraged."
	$(PYTHON_INTERPRETER) -m pip install -q virtualenv virtualenvwrapper
	@echo ">>> Installing virtualenvwrapper if not already installed.\nMake sure the following lines are in shell startup file\n\
	export WORKON_HOME=$$HOME/.virtualenvs\nexport PROJECT_HOME=$$HOME/Devel\nsource /usr/local/bin/virtualenvwrapper.sh\n"
	@bash -c "source `which virtualenvwrapper.sh`;mkvirtualenv $(PROJECT_NAME) --python=$(PYTHON_INTERPRETER)"
	@echo ">>> New virtualenv created. Activate with:\nworkon $(PROJECT_NAME)"
endif

## Test python environment is setup correctly
test_environment:
	$(PYTHON_INTERPRETER) test_environment.py

#################################################################################
# PROJECT RULES                                                                 #
#################################################################################



#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

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
	@sed -n -e "/^## / { \
		h; \
		s/.*//; \
		:doc" \
		-e "H; \
		n; \
		s/^## //; \
		t doc" \
		-e "s/:.*//; \
		G; \
		s/\\n## /---/; \
		s/\\n/ /g; \
		p; \
	}" ${MAKEFILE_LIST} \
	| LC_ALL='C' sort --ignore-case \
	| awk -F '---' \
		-v ncol=$$(tput cols) \
		-v indent=19 \
		-v col_on="$$(tput setaf 6)" \
		-v col_off="$$(tput sgr0)" \
	'{ \
		printf "%s%*s%s ", col_on, -indent, $$1, col_off; \
		n = split($$2, words, " "); \
		line_length = ncol - indent; \
		for (i = 1; i <= n; i++) { \
			line_length -= length(words[i]) + 1; \
			if (line_length <= 0) { \
				line_length = ncol - indent - length(words[i]) - 1; \
				printf "\n%*s ", -indent, " "; \
			} \
			printf "%s ", words[i]; \
		} \
		printf "\n"; \
	}' \
	| more $(shell test $(shell uname) = Darwin && echo '--no-init --raw-control-chars')
