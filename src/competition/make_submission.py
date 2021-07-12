# -*- coding: utf-8 -*-
import logging
import os
import shutil
from pathlib import Path
from src.utils.mask_cleanup import MIN_OBJ_EDGE_ALL, MIN_OBJ_EDGE_ANY, MIN_OBJ_SIZE, MIN_TOTAL, mask_cleanup
from zipfile import ZipFile
import numpy as np

import click
from dotenv import find_dotenv, load_dotenv
from PIL import Image, ImageOps
from scipy import sparse
from tqdm.auto import tqdm


def convert_image(img_path):
    """ Converts a .tif binary mask to a boolean-matrix .npz file
    """
    img = Image.open(img_path)
    return sparse.csr_matrix(ImageOps.invert(img), dtype=bool)


@click.command()
@click.argument('input-filepath', type=click.Path(exists=True))
@click.argument('output-filepath', type=click.Path())
@click.option('-t', '--mask-threshold', default=0.5)
@click.option('-c', '--clean-masks', is_flag=True)
@click.option('-ra', '--replace-aguadas', is_flag=True)
def main(input_filepath, output_filepath, mask_threshold, clean_masks, replace_aguadas):
    """ Runs data processing scripts to turn prediction data into
        a submission ready zip file.
    """

    logger = logging.getLogger(__name__)

    logger.info('Validating files to submit')

    masks = dict()

    npys = Path(input_filepath).glob('*.npy')
    for npy in npys:
        parts = npy.stem.split('_')
        assert len(parts) == 4, \
            f'There should be only 4 parts in the filename, but {len(parts)} were found.'
        assert (1765 <= int(parts[1]) <= 2093), \
            f'File index numbers are not in range'
        assert parts[3] in ['building', 'platform', 'aguada'], \
            f'Mask types are not in accepted list: {parts[3]}'
        masks[parts[3]] = masks.get(parts[3], 0) + 1

    for type in masks:
        assert masks[type] == 329, \
            f'Found {masks[type]} for {type}, but expected 329'

    info = 'Cleaning masks & creating npz files in output_filepath' if clean_masks else 'Creating npz files in output_filepath'
    logger.info(info)

    npys = Path(input_filepath).glob('*.npy')
    for npy in tqdm(npys):
        img = np.load(npy)[0,:,:]
        img = img > mask_threshold
        if clean_masks:
            mask_type = npy.stem.split('_')[3]
            img = mask_cleanup(img, MIN_OBJ_SIZE[mask_type], MIN_OBJ_EDGE_ANY[mask_type], MIN_OBJ_EDGE_ALL[mask_type], MIN_TOTAL[mask_type])
        #Add extra dims which was removed fclean_maskor cleanup
        #images = [ np.expand_dims(im, axis=0) for im in images]
        img = Image.fromarray(img)
        img = sparse.csr_matrix(img, dtype=bool)
        filename = os.path.join(output_filepath, npy.stem + ".npz")
        sparse.save_npz(filename, img)

    if replace_aguadas:
        logger.info('Replacing aguadas')
        aguadas = Path(output_filepath).glob('*aguada.npz')
        for aguada in aguadas:
            filename = aguada.stem
            os.remove(aguada)
            shutil.copyfile(os.path.join("aguadas", f"{filename}.npz"), os.path.join(output_filepath, f"{filename}.npz"))


    logger.info('Creating zip archive and deleting .npz files')

    zipf = ZipFile('submission.zip', 'w')
    npzs = Path(output_filepath).glob('*.npz')
    for npz in npzs:
        #fname = os.path.join(output_filepath, npz)
        zipf.write(npz, os.path.basename(npz))
        os.remove(npz)

    logger.info(f'Created {output_filepath}/submission.zip')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
