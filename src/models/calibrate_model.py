import logging
import os
import sys
from pathlib import Path

import click
import matplotlib.pyplot as plt
import torch
from torch.utils.data.dataloader import DataLoader
from tqdm.auto import tqdm

from src.data.maya_dataset import MayaDataset, MayaTransform
from src.models.deeplabv3 import DeepLabV3
from src.models.unet_adv import UNet as UNetA

from sklearn.calibration import calibration_curve


def predict(net, loader, device, ):
    net.eval()
    n_test = len(loader)

    with tqdm(total=n_test, desc='Inference on test set', unit='batch', leave=False) as pbar:
        preds = []
        mask_building = []
        mask_platform = []
        mask_aguada = []
        for batch in loader:
            lidar = batch['lidar'].to(device=device)
            sentinel2 = batch['sentinel2'].to(device=device)
            mask_building.append(batch['mask_building'])
            mask_platform.append(batch['mask_platform'])
            mask_aguada.append(batch['mask_aguada'])

            with torch.no_grad():
                masks_pred = torch.sigmoid(net(lidar, sentinel2)).cpu()

            preds.append(masks_pred)

            pbar.update()


        mask_building = torch.cat(mask_building, 0).reshape(-1).numpy()
        mask_platform = torch.cat(mask_platform, 0).reshape(-1).numpy()
        mask_aguada = torch.cat(mask_aguada, 0).reshape(-1).numpy()

        preds = torch.cat(preds, 0)
        pred_building = preds[:, 0].reshape(-1).numpy()
        pred_platform = preds[:, 1].reshape(-1).numpy()
        pred_aguada = preds[:, 2].reshape(-1).numpy()

        results_aguada = calibration_curve(mask_aguada, pred_aguada, strategy='uniform', n_bins=10)
        results_building = calibration_curve(mask_building, pred_building, strategy='uniform', n_bins=10)
        results_platform = calibration_curve(mask_platform, pred_platform, strategy='uniform', n_bins=10)

        f, ax = plt.subplots(3)
        ax[0].plot(results_aguada[1], results_aguada[0])
        ax[0].set_title("Aguada calibration")
        ax[1].plot(results_building[1], results_building[0])
        ax[1].set_title("Building calibration")
        ax[2].plot(results_platform[1], results_platform[0])
        ax[2].set_title("Platform calibration")
        plt.show()




@click.command()
@click.option('-mt', '--model_type', default="unet")
@click.option('-m', '--model_path', type=click.Path(exists=True))
@click.option('-i', '--input_dir_root', default=None, type=click.Path(exists=True))
@click.option('-b', '--batch_size', default=1)
def main(model_type, model_path, input_dir_root, batch_size):
    """ Uses the passed model for inference on the files in the input path
    """

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
        input_dir_root, split="train", transform=MayaTransform(
            use_augmentations=False, use_advanced_augmentations=False
        )
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                             num_workers=0, pin_memory=True, drop_last=False)

    try:
        predict(net=net,
                loader=test_loader,
                device=device,
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
