ECML PKDD 2021 - Discovery Challenge
==============================

Our results
-------------
https://docs.google.com/spreadsheets/d/1bsIPCZqQiToaxMSY4HYxHV42kNnmXT5kXaNAT-VrUaM/edit#gid=0


Overview
------------

Explore the potential of the Sentinel satellite data, in combination with the available lidar data, for integrated image segmentation in order to locate and identify “lost” ancient Maya settlements (aguadas, buildings and platforms), hidden under the thick forest canopy.
Check out the [website](https://biasvariancelabs.github.io/maya_challenge/about/) for more info.


<!-- GETTING STARTED -->
Getting Started
------------

1. Clone repo
   ```sh
   git clone https://github.com/thllwg/maya-challenge.git
   ```
2. Change your current working directory to the github repo
   ```sh
   cd maya-challenge
   ```   
3. Pull development container 
   ```sh
   docker pull thllwg/maya-challenge:dev
   ```
4. Run docker image and mount repository into container
   ```sh
   docker run --volume "$(pwd):/workspace/" -p 9000:8888 thllwg/maya-challenge:dev 
   ```
5. You can now open Jupyter Notebook in your browser: [http://0.0.0.0:9000](http://0.0.0.0:9000/).

Downloading data
------------

#### Quickstart

If you don't want to be bothered with raw data but just train a model on preprocessed data already, there is a convenience function available: from your project root:

```sh
make processed_data
```
#### Full Dataset

Otherwise, if you're not only interested in a train_val_test split, but would like to have a look into the raw data too, you can get the full data processing pipeline with:
```sh
# uncomment the following line to remove previously downloaded files
# make clean
make data
```
The `make data` command downloads the challenge files to `data/raw`, produces `.npy` files for all images in `data/interim` and a train_val_test split in `data/processed`. 


Project Organization
------------

    ├── LICENSE            <- NO LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make predict_ensemble`
    ├── README.md          <- The top-level README documenting our approach.
    ├── data
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump. Use `make data` to load data set.
    │
    ├── runs
    │   └── unet_bce_6e-...<- A specific run directory with model type, parameters and timestamp
    │       ├── model      <- Trained and serialized models, we store every best model
    │       ├── tensorboard<- Tensorboard logs
    │       └── logs.txt   <- Logs written during training
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── predictions        <- Model predictions
    │
    ├── notebooks          <- Jupyter notebooks. 
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── environment.yml   <- Environment file for reproducing the analysis environment with conda. Use this if you don't want to use our docker container
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── competition    <- Scripts to create a submission file in the specified format
    │   │   └── make_submission.py
    │   ├── data           <- Scripts to process raw data or to perform a test-train-split on file level
    │   │   ├── maya_dataset.py
    │   │   └── make_dataset.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── ensemble_predictions.py
    │   │   ├── predict_model.py
    │   │   ├── eval_model.py
    │   │   └── train_model.py
    │   │
    │   ├── utils          <- Scripts to clean up predicted or ground truth masks 
    │   │   │
    │   │   ├── board_logger.py
    │   │   ├── mask_cleanup.py
    │   │   └── voting.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

