import argparse
import logging
from os import listdir
from os.path import join, isdir
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from tqdm.auto import tqdm

from src.data.maya_dataset import MayaDataset, MayaTransform


class EnsembleDataset(Dataset):
    def __init__(self, root_dir, ignore_folder):
        self.root_dir = root_dir
        self.label_sources = {}

        for model in listdir(root_dir):
            model_dir = join(root_dir, model)
            if not isdir(model_dir) or model == ignore_folder:
                continue
            self.label_sources[model] = {}

            for file in listdir(model_dir):
                num = int(file.split("_")[1])
                cls = file.split("_")[-1].split(".")[0]
                if num not in self.label_sources[model]:
                    self.label_sources[model][num] = {}
                self.label_sources[model][num][cls] = join(model_dir, file)
            self.item_count = len(self.label_sources[model].keys())

        self.n_models = len(self.label_sources)

    def __getitem__(self, idx):
        preds = []
        for model_key in self.label_sources:
            model_sources = self.label_sources[model_key]
            preds.append(np.concatenate([
                np.load(model_sources[idx]["building"]),
                np.load(model_sources[idx]["platform"]),
                np.load(model_sources[idx]["aguada"])
            ], 0))
        preds = np.stack(preds, 0)
        return torch.tensor(preds)

    def __len__(self):
        return self.item_count


class CombineDataset(Dataset):
    def __init__(self, test_dataset, ensemble_dataset):
        self.test_dataset = test_dataset
        self.ensemble_dataset = ensemble_dataset

    def __getitem__(self, idx):
        test = self.test_dataset[idx]
        mask_building = test["mask_building"]
        mask_platform = test["mask_platform"]
        mask_aguada = test["mask_aguada"]

        return mask_building, mask_platform, mask_aguada, self.ensemble_dataset[idx]

    def __len__(self):
        return len(self.test_dataset)


class Weighter(nn.Module):
    def __init__(self, n_models, n_classes, model_names):
        super().__init__()
        self.weights = nn.Parameter(
            torch.torch.randn(1, n_models, n_classes, 1, 1) * 0.01 + 1 / n_models, requires_grad=True
        )
        self.bias = nn.Parameter(torch.zeros(1, 1, n_classes, 1, 1), requires_grad=True)
        self.model_names = model_names

    def forward(self, x):
        x = torch.sigmoid((x * self.weights + self.bias).sum(1))

        return x



def get_args():
    parser = argparse.ArgumentParser(description='Ensemble predict on images',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-i', '--input_dir', type=str, default='./data/processed/',
                        help='Dir with the images')
    parser.add_argument('-pe', '--prediction_dir', default='./ensemble_weighting',
                        help='Path where output of the model is cached. Use the folder name from run after this base path')
    parser.add_argument('-e', '--ensemble_dir', default='ensemble', help='Ensemble dir inside the output dir')
    parser.add_argument('-b', '--batch_size', metavar='B', type=int, default=8, help='Batch size')
    parser.add_argument('--epochs', type=int, default=10, help='Number of Epochs')
    parser.add_argument('-s', '--dataset_split', type=str, default='train',
                        help='Split on which the predictions are performed')
    parser.add_argument('-t', '--threshold', type=float, help='Threshold for selecting the mask', default=0.5)
    parser.add_argument('-nw', '--num_workers', type=int, default=0, help='how many data workers per dataloader')
    parser.add_argument('-l', '--learning-rate', metavar='LR', type=float, default=0.01,
                        help='Learning rate')
    return parser.parse_args()


# def collate_fn(data_batch):
#     import ipdb;
#     ipdb;
#     ipdb.set_trace()
#     mask_building = []
#     mask_platform = []
#     mask_aguada = []
#     ensemble = []
#     for data in data_batch:
#         mask_building.append(data[0])
#         mask_platform.append(data[1])
#         mask_aguada.append(data[2])
#         ensemble.append(data[3])
#     mask_building = torch.stack(mask_building, 0)
#     mask_platform = torch.stack(mask_platform, 0)
#     mask_aguada = torch.stack(mask_aguada, 0)
#     ensemble = torch.stack(ensemble, 0)
#     return mask_building, mask_platform, mask_aguada, ensemble


if __name__ == '__main__':
    args = get_args()

    output_dir = Path(args.prediction_dir)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    logs_file = output_dir / 'ensemble_weighting_logs.txt'
    logging.basicConfig(filename=logs_file, level=logging.INFO, format='%(levelname)s: %(message)s')
    logging.info(f'Writing the ensemble logs to {logs_file}')

    test_dataset = MayaDataset(
        args.input_dir, split="train", transform=MayaTransform(
            use_augmentations=False, use_advanced_augmentations=False
        )
    )
    ensemble_dataset = EnsembleDataset(args.prediction_dir, args.ensemble_dir)
    model_names = list(ensemble_dataset.label_sources.keys())

    dataset = CombineDataset(test_dataset, ensemble_dataset)
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                             num_workers=args.num_workers, pin_memory=True, drop_last=True)

    n_models = len(ensemble_dataset.label_sources)
    n_classes = 3

    weighter = Weighter(n_models, n_classes, model_names)
    weighter.to(device)

    optim = torch.optim.AdamW(weighter.parameters(), lr=args.learning_rate)
    criterion = nn.BCELoss() # because we directly use probs

    for epoch in range(args.epochs):
        pbar = tqdm(data_loader)
        for mask_building, mask_platform, mask_aguada, ens_preds in pbar:
            if mask_building.sum() == mask_platform.sum() == mask_aguada.sum() == 0:
                continue
            optim.zero_grad()

            mask_building = mask_building.to(device=device, dtype=torch.float32)
            mask_platform = mask_platform.to(device=device, dtype=torch.float32)
            mask_aguada = mask_aguada.to(device=device, dtype=torch.float32)
            ens_preds = ens_preds.to(device)

            ens_preds = weighter(ens_preds)

            b = criterion(ens_preds[:, [0]], mask_building)
            p = criterion(ens_preds[:, [1]], mask_platform)
            a = criterion(ens_preds[:, [2]], mask_aguada)

            tloss = (b + p + a) / 3

            tloss.backward()

            optim.step()
            pbar.set_postfix_str(
                f"loss: {tloss.item(): 0.4f}; b: {b.item(): 0.4f}; p: {p.item(): 0.4f}; a: {a.item(): 0.4f}"
            )

    dict = {}
    dict["weights"] = weighter.weights[0, :, :, 0, 0].data.cpu()
    dict["bias"] = weighter.bias[0, :, :, 0, 0].data.cpu()
    dict["models"] = weighter.model_names
    torch.save(dict, join(args.prediction_dir, "weights.pth"))

    logging.info(f'''
        Model names and idx: {[f"{i}: {name}" for i, name in enumerate(weighter.model_names)]}
    
        Building:
            Weights: {str(dict["weights"][:, 0])}
            Bias:    {str(dict["bias"][:, 0])}
        Platform:
            Weights: {str(dict["weights"][:, 1])}
            Bias:    {str(dict["bias"][:, 1])}
        Aguada:
            Weights: {str(dict["weights"][:, 2])}
            Bias:    {str(dict["bias"][:, 2])}
    ''')
