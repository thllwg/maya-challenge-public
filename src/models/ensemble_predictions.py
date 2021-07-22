import argparse
import logging
import os
import shutil
from pathlib import Path
from typing import List

import numpy as np
import torch
from torch.utils.data.dataloader import DataLoader

from src.data.maya_dataset import MayaDataset, MayaTransform
from src.models import str2bool
from src.models.deeplabv3 import DeepLabV3
from src.models.predict_model import predict
from src.models.unet_adv import UNet as UNetA
from src.utils.voting import majority_voting


def predicted_path_check(predicted_paths: List[Path]):
    logging.info(f'Verifying that {predicted_paths} have the same files.')

    cm = None
    for p in predicted_paths:
        f = sorted(p.glob('*.npy'))
        f = [n.name for n in f]
        assert len(f) > 0
        if cm is not None:
            assert cm == f
        else:
            cm = f


def voting_predictions(predicted_paths: List[Path], output_dir: Path, voting: str = 'soft', weights=None,
                         threshold=0.5):
    logging.info(f'''
    Paths to ensemble:      {predicted_paths}
    Write ensemble to:      {output_dir}
    Voting:                 {voting}
    Weights:                {weights}
    Threshold:              {threshold}
    ''')

    if output_dir.exists():
        logging.warning(f'Output directory already exists. Cleaning the {output_dir} now!!!')
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    logging.info(f'Writing the ensemble output to {output_dir}')

    files_to_ensemble = sorted(predicted_paths[0].glob('*.npy'))
    files_to_ensemble = [n.name for n in files_to_ensemble]

    for filename in files_to_ensemble:
        masks = []

        for p in predicted_paths:

            mask_file = p / filename
            img = np.load(mask_file)
            masks.append(img)

        masks = np.stack(masks, axis=0) 
        ensembled_mask = majority_voting(masks, voting, weights, threshold)
        ofn = output_dir / filename
        np.save(ofn, ensembled_mask)

    logging.info(f'Ensemble complete, written to {output_dir}')

    predicted_paths.append(output_dir)
    predicted_path_check(predicted_paths)


def ensemble_predictions(predicted_paths: List[Path], output_dir: Path, threshold=0.5):
    logging.info(f'''
    Paths to ensemble:      {predicted_paths}
    Write ensemble to:      {output_dir}
    Threshold:              {threshold}
    ''')

    if output_dir.exists():
        logging.warning(f'Output directory already exists. Cleaning the {output_dir} now!!!')
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    logging.info(f'Writing the ensemble output to {output_dir}')

    all_weights = {
        "aguada": [0.1, 0.4, 0.5],
        "aguada_bias": 1,
        "building": [0.1, 0.4, 0.5],
        "building_bias": 1,
        "platform": [0.1, 0.4, 0.5],
        "platform_bias": 1,
    }

    files_to_ensemble = sorted(predicted_paths[0].glob('*.npy'))
    files_to_ensemble = [n.name for n in files_to_ensemble]
    for filename in files_to_ensemble:
        masks = []
        weighted_masks = []
        weights = np.array(all_weights[filename.split(".")[0].split("_")[3]])
        bias = np.array(all_weights[filename.split(".")[0].split("_")[3]+"_bias"])

        for i, p in enumerate(predicted_paths):

            mask_file = p / filename
            img = np.load(mask_file)
            weighted_img = img * weights[i]
            weighted_masks.append(weighted_img)
            masks.append(img)
            
        weighted_masks = np.stack(weighted_masks, axis=-1)
        masks = np.stack(masks, axis=-1) 

        masks_reshaped = masks.reshape(-1,len(predicted_paths)) 
        assert np.all(np.sum(masks_reshaped, axis=0) == np.sum(masks, axis = (0,1,2)))

        masks_reshaped = masks_reshaped * weights

        assert np.allclose(masks_reshaped.reshape(weighted_masks[0].shape), weighted_masks[0])

        masks = masks * weights

        assert np.allclose(masks, weighted_masks)

        masks = np.sum(masks, axis=-1) + bias
        ofn = output_dir / filename
        np.save(ofn, masks)

    logging.info(f'Ensemble complete, written to {output_dir}')

    predicted_paths.append(output_dir)
    predicted_path_check(predicted_paths)



def bulk_predict(models, input_dir, output_dir, batch_size, test_augmentation, dataset_split='test'):
    predicted_paths = []

    for m in models:
        model = PretrainedModel(m)
        model.path = Path(model.path)
        model_name = model_name_from_path(model.path)

        if model.path.is_dir():
            model.path = model.path / "BestModel.pth"

        if not model.path.is_file():
            raise ValueError(f"No model at location: {model.path}")
        logging.info(f'Predicting with model {model.path}.')

        input_dir = Path(input_dir)
        model_output_dir = Path(output_dir) / model_name
        logging.info(f'Writing predictions of {model.path} to {model_output_dir}')

        if model_output_dir.exists():
            logging.warning(f'Output directory already exists. Cleaning the {model_output_dir} now!!!')
            shutil.rmtree(model_output_dir)

        model_output_dir.mkdir(parents=True, exist_ok=False)
        predicted_paths.append(model_output_dir)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logging.info(f'Using device {device}')
        if model.type == "deeplabv3":
            net = DeepLabV3.load_model(model.path, device)
        elif model.type == "unet_adv":
            net = UNetA.load_model(model.path, device)
        else:
            raise Exception(f'Unknowng model type: {model.type}')

        test_dataset = MayaDataset(
            input_dir, split=dataset_split, transform=MayaTransform(
                use_augmentations=False, use_advanced_augmentations=False
            )
        )
        test_loader = DataLoader(test_dataset, batch_size=batch_size,
                                 num_workers=0, pin_memory=True, drop_last=False)

        predict(net=net,
                loader=test_loader,
                device=device,
                output_dir=model_output_dir,
                test_augmentation=test_augmentation
                )

    predicted_path_check(predicted_paths)

    return predicted_paths


class PretrainedModel():
    def __init__(self, type_path) -> None:
        type_path = type_path[0]
        if not type_path.endswith(os.path.join("model", "BestModel.pth")):
            if not type_path.endswith(".pth"):
                type_path = os.path.join(type_path, "model", "BestModel.pth")
        for t in ["unet_adv", "deeplabv3", "unet"]:
            if t in type_path:
                self.type = t
                self.path = type_path
                break
        if not self.type:
            ValueError('Unrecognized model type')


def model_name_from_path(p):
    if 'runs' in p.parts:
        i = p.parts.index('runs')
        if i + 1 == len(p.parts):
            ValueError('The model is expected to be saved in it saved in its own dir')
        else:
            return p.parts[i + 1]
    elif p.is_dir():
        return p.parts[-1]
    else:
        NotImplemented('Could not handle the incoming path')


def get_args():

    parser = argparse.ArgumentParser(description='Ensemble predict on images',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     fromfile_prefix_chars='@')
    parser.add_argument('-i', '--input_dir', type=str, default='./data/processed/',
                        help='Dir with the images')
    parser.add_argument('-m', '--models', action='append', nargs=1,
                        metavar=('type_path'),
                        help='E.g. -m unet bce_June26_001  -m deeplabv3 unet_June29. This option can be repeated.')
    parser.add_argument('-o', '--output_dir', default='./predictions',
                        help='Path where output of the model is cached. Use the folder name from run after this base path')
    parser.add_argument('-e', '--ensemble_dir', default='ensemble', help='Ensemble dir inside the output dir')
    parser.add_argument('-b', '--batch_size', metavar='B', type=int, default=8, help='Batch size')
    parser.add_argument('-s', '--dataset_split', type=str, default='test',
                        help='Split on which the predictions are performed')
    parser.add_argument('-v', '--voting', type=str, default='soft', help='soft or hard voting')
    parser.add_argument('-w', '--weights', nargs='*', type=int,
                        help='Weights for each model to be ensembled, eg. -w 1 2 1 1 4 ...', default=None)
    parser.add_argument('-t', '--threshold', type=float, help='Threshold for selecting the mask', default=0.5)
    parser.add_argument('-ta', '--test-augmentation', type=str2bool, default=False,
                        help="Whether to use test augmentations (flips and 90° rotations).")
    parser.add_argument('-en', '--ensemble', type=str2bool, default=False,
                        help="")
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logs_file = output_dir / 'ensemble_logs.txt'
    logging.basicConfig(filename=logs_file, level=logging.INFO, format='%(levelname)s: %(message)s')
    logging.info(f'Writing the ensemble logs to {logs_file}')

    predicted_paths = bulk_predict(
        args.models, args.input_dir, args.output_dir, args.batch_size, args.test_augmentation, args.dataset_split
    )

    
    #predicted_paths = [Path(os.path.join(output_dir, x)) for x in os.listdir(output_dir) if x != "ensemble_logs.txt"]
    #print(predicted_paths)

    if args.ensemble:
        ensemble_output_path = output_dir / args.ensemble_dir
        ensemble_predictions(predicted_paths, ensemble_output_path, args.threshold)
    else:
        ensemble_output_path = output_dir / args.ensemble_dir
        voting_predictions(predicted_paths, ensemble_output_path, args.voting, args.weights, args.threshold)
