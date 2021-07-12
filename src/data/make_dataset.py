# -*- coding: utf-8 -*-
import click
import logging
import os
from click.types import BOOL
import numpy as np
import rasterio
import random
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
from tqdm import tqdm
import shutil

def split_maya_train(interim_data_dir = '../../data/interim', processed_data_dir = '../../data/processed', val_percent = 0.1, overwrite = False, seed = 4357631117):
    samples = list(range(1, 1764))
    n_val = int(len(samples) * val_percent)
    random.Random(seed).shuffle(samples)
    val_samples = samples[:n_val]
    
    # Step 1: Copy all the interim dir + files to the processed
    shutil.copytree(interim_data_dir, processed_data_dir) 
    #os.system(f"cp -rf {interim_data_dir + '/*'} {processed_data_dir}")

    # Step 2: Get list of val dirs - training directories
    train_dirs = os.listdir(interim_data_dir)
    train_dirs = [i for i in train_dirs if 'train' in i]
    print(train_dirs)
    val_dirs = [i.replace('train', 'val') for i in train_dirs]

    # Make the val dirs
    for v in val_dirs:
        os.makedirs( os.path.join(processed_data_dir, v), exist_ok=True)

    # Move the val samples to the val dirs
    for tr in train_dirs:
        fls = os.listdir(os.path.join(processed_data_dir, tr))
        for f in fls:
            n = f.split('_')[1]
            if int(n) in val_samples:
                os.rename( os.path.join(processed_data_dir, tr, f), os.path.join(processed_data_dir, tr.replace('train', 'val'), f))


@click.command()
@click.argument('raw_data_dir', type=click.Path(exists=True))
@click.argument('interim_data_dir', type=click.Path())
@click.argument('processed_data_dir', type=click.Path())
@click.option('--val_percent', default=0.1, type=float)
@click.option('--overwrite', default=True, type=bool)
def main(raw_data_dir, interim_data_dir, processed_data_dir, val_percent, overwrite):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('Creating .npy files from raw data')

    subdirs = [f.name for f in os.scandir(raw_data_dir) if f.is_dir()]

    for dir in tqdm(subdirs):

        processed_path = os.path.join(interim_data_dir, dir)
        Path(processed_path).mkdir(parents=True, exist_ok=True)

        for entry in os.scandir(os.path.join(raw_data_dir, dir)):
            if entry.name.endswith(".tif") or entry.name.endswith(".tiff"):
                name = Path(entry.path).stem+".npy"
                with rasterio.open(Path(entry.path), "r") as im:
                    np.save(os.path.join(processed_path, name), im.read())

    os.rename(os.path.join(interim_data_dir,"train_masks"), os.path.join(interim_data_dir, "masks_train"))

    if overwrite == True:
        shutil.rmtree(processed_data_dir)

    split_maya_train(interim_data_dir = interim_data_dir, processed_data_dir = processed_data_dir, val_percent = val_percent, overwrite = overwrite)



if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
