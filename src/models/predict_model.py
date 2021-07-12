import logging
import os
import sys
from pathlib import Path

import click
import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageOps
from scipy import sparse
from torch.utils.data.dataloader import DataLoader
from tqdm.auto import tqdm

from src.data.maya_dataset import MayaDataset, MayaTransform
from src.models import str2bool
from src.models.deeplabv3 import DeepLabV3
from src.models.unet import UNet
from src.models.unet_adv import UNet as UNetA


def predict(net,
            loader,
            device,
            output_dir='./data/predictions',
            img_type=torch.float32,
            test_augmentation=False,
            ):
    net.eval()
    n_test = len(loader)
    sigmoid = nn.Sigmoid()
    with tqdm(total=n_test, desc='Inference on test set', unit='batch', leave=False) as pbar:
        for batch in loader:
            lidar = batch['lidar'].to(device=device, dtype=img_type)
            sentinel2 = batch['sentinel2'].to(device=device, dtype=img_type)

            with torch.no_grad():
                masks_pred = sigmoid(net(lidar, sentinel2))

            if test_augmentation:
                with torch.no_grad():
                    # wflip, hflip, flip
                    masks_pred += sigmoid(net(lidar.flip(2), sentinel2.flip(2)).flip(2))
                    masks_pred += sigmoid(net(lidar.flip(3), sentinel2.flip(3)).flip(3))
                    masks_pred += sigmoid(net(lidar.flip(2).flip(3), sentinel2.flip(2).flip(3)).flip(3).flip(2))

                    # rotate
                    masks_pred += sigmoid(net(lidar.rot90(1, [2, 3]), sentinel2.rot90(1, [2, 3])).rot90(-1, [2, 3]))
                    masks_pred += sigmoid(net(lidar.rot90(2, [2, 3]), sentinel2.rot90(2, [2, 3])).rot90(-2, [2, 3]))
                    masks_pred += sigmoid(net(lidar.rot90(3, [2, 3]), sentinel2.rot90(3, [2, 3])).rot90(-3, [2, 3]))

                    # hflip then rotate
                    masks_pred += sigmoid(net(
                        lidar.flip(2).rot90(1, [2, 3]), sentinel2.flip(2).rot90(1, [2, 3])
                    ).rot90(-1, [2, 3]).flip(2))
                    masks_pred += sigmoid(net(
                        lidar.flip(2).rot90(2, [2, 3]), sentinel2.flip(2).rot90(2, [2, 3])
                    ).rot90(-2, [2, 3]).flip(2))
                    masks_pred += sigmoid(net(
                        lidar.flip(2).rot90(3, [2, 3]), sentinel2.flip(2).rot90(3, [2, 3])
                    ).rot90(-3, [2, 3]).flip(2))

                    # vflip then rotate
                    masks_pred += sigmoid(net(
                        lidar.flip(3).rot90(1, [2, 3]), sentinel2.flip(3).rot90(1, [2, 3])
                    ).rot90(-1, [2, 3]).flip(3))
                    masks_pred += sigmoid(net(
                        lidar.flip(3).rot90(2, [2, 3]), sentinel2.flip(3).rot90(2, [2, 3])
                    ).rot90(-2, [2, 3]).flip(3))
                    masks_pred += sigmoid(net(
                        lidar.flip(3).rot90(3, [2, 3]), sentinel2.flip(3).rot90(3, [2, 3])
                    ).rot90(-3, [2, 3]).flip(3))

                    # flip then rotate
                    masks_pred += sigmoid(net(
                        lidar.flip(2).flip(3).rot90(1, [2, 3]), sentinel2.flip(2).flip(3).rot90(1, [2, 3])
                    ).rot90(-1, [2, 3]).flip(3).flip(2))
                    masks_pred += sigmoid(net(
                        lidar.flip(2).flip(3).rot90(2, [2, 3]), sentinel2.flip(2).flip(3).rot90(2, [2, 3])
                    ).rot90(-2, [2, 3]).flip(3).flip(2))
                    masks_pred += sigmoid(net(
                        lidar.flip(2).flip(3).rot90(3, [2, 3]), sentinel2.flip(2).flip(3).rot90(3, [2, 3])
                    ).rot90(-3, [2, 3]).flip(3).flip(2))

                    masks_pred /= 16

            masks = masks_pred

            for i in range(len(batch['idx'])):
                filenames = [os.path.join(output_dir, f'tile_{batch["idx"][i].item()}_mask_{m}.npy') for m in [
                    "building", "platform", "aguada"]]
                images = [(masks[i, [j]].cpu().numpy()).astype(np.float32) for j in range(3)]
                [np.save(filename, image) for filename, image in zip(filenames, images)]

            pbar.update()


def mask_to_image(mask):
    return Image.fromarray((mask * 255).astype(np.uint8))


# Converts binary mask to a boolean-matrix .npz file for submission
def convert_image(img):
    return sparse.csr_matrix(ImageOps.invert(img), dtype=bool)


@click.command()
@click.option('-mt', '--model_type', default="unet")
@click.option('-m', '--model_path', type=click.Path(exists=True))
@click.option('-i', '--input_dir_root', default=None, type=click.Path(exists=True))
@click.option('-o', '--output_dir', default=None, type=click.Path(exists=True))
@click.option('-b', '--batch_size', default=1)
@click.option('-ta', '--test-augmentation', is_flag=True)
def main(model_type, model_path, input_dir_root, output_dir, batch_size, test_augmentation):
    """ Uses the passed model for inference on the files in the input path
    """
    test_augmentation = str2bool(test_augmentation)

    logging.getLogger(__name__)

    model_path = Path(model_path)
    if model_path.is_dir():
        model_path = model_path / "BestModel.pth"
    if not model_path.is_file():
        raise ValueError("No model at location.")

    # checking default locations for input and output files
    project_dir = Path(__file__).resolve().parents[2]
    if input_dir_root is None:
        input_dir_root = project_dir / "data" / "processed" / "lidar_test"
    else:
        input_dir_root = Path(input_dir_root)

    if output_dir is None:
        output_dir = project_dir / "predictions"
    else:
        output_dir = Path(output_dir)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    if model_type == "unet":
        net = UNet.load_model(model_path, device)
    elif model_type == "deeplabv3":
        net = DeepLabV3.load_model(model_path, device)
    elif model_type == "unet_adv":
        net = UNetA.load_model(model_path, device)
    else:
        raise Exception(f'Unknowng model type: {model_type}')

    test_dataset = MayaDataset(
        input_dir_root, split="test", transform=MayaTransform(
            use_augmentations=False, use_advanced_augmentations=False
        )
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                             num_workers=0, pin_memory=True, drop_last=False)

    try:
        predict(net=net,
                loader=test_loader,
                device=device,
                output_dir=output_dir,
                test_augmentation=test_augmentation
                )
    except KeyboardInterrupt:
        logging.info('Prediction Interrupted!')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    project_dir = Path(__file__).resolve().parents[2]

    main()
