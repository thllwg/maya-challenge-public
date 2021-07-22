import logging
import os
import random

import numpy as np
import torch
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset, Subset, WeightedRandomSampler
from torchvision.transforms import RandomCrop
from src.utils.mask_cleanup import mask_cleanup, MIN_OBJ_SIZE, MIN_OBJ_EDGE_ANY, MIN_OBJ_EDGE_ALL, MIN_TOTAL

random.seed(18734857687)


class MayaTransform():
    def __init__(self, use_augmentations: bool, use_advanced_augmentations: bool = False, crop_size: int = 400) -> None:
        self.use_augmentations = use_augmentations
        self.use_advanced_augmentations = use_advanced_augmentations
        self.crop_size = crop_size

    def __call__(self, sample, *args, **kwds) -> object:
        """
        Given the variabiltiy in the data source the transformation are written in pythonically ( explicity, and sequentially).
        It is possible to use image augmentations libraries here, if they provide an functional interface
        """

        lidar = torch.FloatTensor(sample['lidar'])

        sentinel1 = torch.FloatTensor(sample['sentinel1'])
        sentinel2 = torch.FloatTensor(sample['sentinel2'])
        mask_building = torch.FloatTensor(sample['mask_building'])
        mask_platform = torch.FloatTensor(sample['mask_platform'])
        mask_aguada = torch.FloatTensor(sample['mask_aguada'])

        # See https://github.com/thllwg/maya-challenge/issues/36 and notebooks/ChannelHistogramVisualisation.ipynb for a more detailed analysis
        # Note: These values are after the median cloud composite!!
        sentinel2_stats = {
            'train': {
                'min': np.array([0, 0., 0, 0], dtype=np.float32),
                'max': np.array([0.12078185, 0.10393997, 0.08051628, 0.345], dtype=np.float32),
                'mean': np.array([0.03848704, 0.05012547, 0.03123818, 0.24411583], dtype=np.float32),
                'median': np.array([0.04067905, 0.0541, 0.03341332, 0.2666], dtype=np.float32),
                'q.25': np.array([0.03695997, 0.05050562, 0.03040081, 0.25400001], dtype=np.float32),
                'q.75': np.array([0.04449117, 0.05736133, 0.03626735, 0.278], dtype=np.float32),
                'q.90': np.array([0.04888396, 0.06068037, 0.03893332, 0.28780001], dtype=np.float32),
            }
        }

        # Start by basic normalization (range -> [0-1])
        lidar = lidar / 255.0

        # create median composites
        sentinel2 = sentinel2.view(-1, 13, *sentinel2.shape[1:])  # time, channel, width, height

        ''' mean version (usually median is used) 
        no_clouds = 1 - sentinel2[:, [-1]]
        rgb = sentinel2[:, [3, 2, 1]] * no_clouds
        nir = sentinel2[:, [7]] * no_clouds
        rgb = rgb.sum(0) / no_clouds.sum(0)
        nir = nir.sum(0) / no_clouds.sum(0)
        sentinel2 = torch.cat([rgb, nir], 0)
        '''

        # easiest way to get correct median
        clouds = sentinel2[:, [-1]].bool().numpy()

        rgb_nir = sentinel2[:, [3, 2, 1, 7]].numpy()
        rgb_nir[clouds.repeat(4, axis=1)] = np.nan
        sentinel2 = torch.tensor(np.nanmedian(rgb_nir, 0))

        # Broadcasting starts with the trailing (i.e. rightmost) dimensions and works its way left, therefore we move the channel dimension to the right
        # We use min and 0.90 quantile for normalization
        mn = torch.tensor(sentinel2_stats['train']['min'])
        mx = torch.tensor(sentinel2_stats['train']['q.90'])
        sentinel2 = (sentinel2.permute(1, 2, 0) - mn) / (mx - mn)
        sentinel2 = sentinel2.permute(2, 0, 1)

        mask_building = 1 - (mask_building / 255.0)
        mask_platform = 1 - (mask_platform / 255.0)
        mask_aguada = 1 - (mask_aguada / 255.0)

        # Start of image transformations
        if self.use_augmentations:
            # random cropping (always applied)
            i, j, h, w = RandomCrop.get_params(lidar, output_size=(self.crop_size, self.crop_size))
            lidar = TF.crop(lidar, i, j, h, w)
            sentinel1 = TF.crop(sentinel1, i // 20, j // 20, h // 20, w // 20)
            sentinel2 = TF.crop(sentinel2, i // 20, j // 20, h // 20, w // 20)
            mask_building = TF.crop(mask_building, i, j, h, w)
            mask_platform = TF.crop(mask_platform, i, j, h, w)
            mask_aguada = TF.crop(mask_aguada, i, j, h, w)

            # vertical flip
            if random.random() > 0.5:
                lidar = TF.vflip(lidar)
                sentinel1 = TF.vflip(sentinel1)
                sentinel2 = TF.vflip(sentinel2)
                mask_building = TF.vflip(mask_building)
                mask_platform = TF.vflip(mask_platform)
                mask_aguada = TF.vflip(mask_aguada)

            # horizontal flip
            if random.random() > 0.5:
                lidar = TF.hflip(lidar)
                sentinel1 = TF.hflip(sentinel1)
                sentinel2 = TF.hflip(sentinel2)
                mask_building = TF.hflip(mask_building)
                mask_platform = TF.hflip(mask_platform)
                mask_aguada = TF.hflip(mask_aguada)

            if self.use_advanced_augmentations:
                # rotations
                if random.random() > 0.25:
                    r = np.random.uniform(low=0, high=360, size=1)[0]
                    lidar = TF.rotate(lidar, r)
                    sentinel1 = TF.rotate(sentinel1, r)
                    sentinel2 = TF.rotate(sentinel2, r)
                    mask_building = TF.rotate(mask_building, r)
                    mask_platform = TF.rotate(mask_platform, r)
                    mask_aguada = TF.rotate(mask_aguada, r)

                    # brightness, saturation, and contrast change
                    # # lidar
                    # min_val = 0.8
                    # max_val = 1.2
                    # if random.random() > 0.75:
                    #     sigma = random.random() * (max_val - min_val) + min_val
                    #     lidar = TF.adjust_brightness(lidar, sigma)
                    # if random.random() > 0.75:
                    #     sigma = random.random() * (max_val - min_val) + min_val
                    #     lidar = TF.adjust_saturation(lidar, sigma)
                    # if random.random() > 0.75:
                    #     sigma = random.random() * (max_val - min_val) + min_val
                    #     lidar = TF.adjust_contrast(lidar, sigma)

                    # # sentinel 1
                    # min_val = 0.8
                    # max_val = 1.2
                    # if random.random() > 0.75:
                    #     sigma = random.random() * (max_val - min_val) + min_val
                    #     sentinel1 = TF.adjust_brightness(sentinel1, sigma)
                    # if random.random() > 0.75:
                    #     sigma = random.random() * (max_val - min_val) + min_val
                    #     sentinel1 = TF.adjust_saturation(sentinel1, sigma)
                    # if random.random() > 0.75:
                    #     sigma = random.random() * (max_val - min_val) + min_val
                    #     sentinel1 = TF.adjust_contrast(sentinel1, sigma)

                # # sentinel 2
                min_val = 0.8
                max_val = 1.2
                if random.random() > 0.75:
                    sigma = random.random() * (max_val - min_val) + min_val
                    sentinel2[[0, 1, 2]] = TF.adjust_brightness(sentinel2[[0, 1, 2]], sigma)
                if random.random() > 0.75:
                    sigma = random.random() * (max_val - min_val) + min_val
                    sentinel2[[0, 1, 2]] = TF.adjust_saturation(sentinel2[[0, 1, 2]], sigma)
                if random.random() > 0.75:
                    sigma = random.random() * (max_val - min_val) + min_val
                    sentinel2[[0, 1, 2]] = TF.adjust_contrast(sentinel2[[0, 1, 2]], sigma)

                # Gaussian blur
                if random.random() > 0.75:
                    kernel_size = 11
                    min_val = 0.1
                    max_val = 2
                    sigma = random.random() * (max_val - min_val) + min_val
                    lidar = TF.gaussian_blur(lidar, kernel_size, sigma)

                ## lidar noise
                r = random.random()
                if r < .125:
                    # normal noise
                    sigma = .03
                    lidar = lidar + torch.randn_like(lidar).abs() * sigma * -(lidar - .5).sign()
                elif r < .25:
                    # uniform noise
                    sigma = .1
                    lidar = lidar + torch.rand_like(lidar) * sigma * -(lidar - .5).sign()

                ## sentinel1 noise
                r = random.random()
                if r < .125:
                    # normal noise
                    sigma = .03
                    sentinel1 = sentinel1 + torch.randn_like(sentinel1).abs() * sigma * -(sentinel1 - .5).sign()
                elif r < .25:
                    # uniform noise
                    sigma = .1
                    sentinel1 = sentinel1 + torch.rand_like(sentinel1) * sigma * -(sentinel1 - .5).sign()

                ## sentinel2 noise
                r = random.random()
                if r < .125:
                    # normal noise
                    sigma = .03
                    sentinel2 = sentinel2 + torch.randn_like(sentinel2).abs() * sigma * -(sentinel2 - .5).sign()
                elif r < .25:
                    # uniform noise
                    sigma = .1
                    sentinel2 = sentinel2 + torch.rand_like(sentinel2) * sigma * -(sentinel2 - .5).sign()
                    # End of image transformations

                    # assert lidar.min() >= 0.0 and lidar.max() <= 1.0 \
                    #     and ((mask_building==0.0) | (mask_building==1.0)).all() \
                    #     and ((mask_platform==0.0) | (mask_platform==1.0)).all() \
                    #     and ((mask_aguada==0.0) | (mask_aguada==1.0)).all(),    \
                    #     f'The lidar data or the annotations have incorrect range!!'

                    # Some augmentations add negative strides to the numpy array which leads to problems in them converting to tensors, to remove those we use arr.copy()
                    # See: https://discuss.pytorch.org/t/torch-from-numpy-not-support-negative-strides/3663

        # Some augmentations add negative strides to the numpy array which leads to problems in them converting to tensors, to remove those we use arr.copy()
        # See: https://discuss.pytorch.org/t/torch-from-numpy-not-support-negative-strides/3663
        transformed_sample = {
            'idx': sample['idx'],
            'ori_lidar': sample['lidar'],
            'ori_sentinel1': sample['sentinel1'],
            'ori_sentinel2': sample['sentinel2'],
            'ori_mask_building': sample['mask_building'],
            'ori_mask_platform': sample['mask_platform'],
            'ori_mask_aguada': sample['mask_aguada'],
            'lidar': lidar,
            'sentinel1': sentinel1,
            'sentinel2': sentinel2,
            'mask_building': mask_building,
            'mask_platform': mask_platform,
            'mask_aguada': mask_aguada
        }
        return transformed_sample


class MayaDataset(Dataset):
    def __init__(self, imgs_dir_root, split='train', transform=None, clean_masks=False):
        self.imgs_dir_root = imgs_dir_root
        self.split = split
        self.transform = transform
        self.clean_masks = clean_masks
        self.data_info = {
            'lidar': {
                'dir': 'lidar_',
                'image_suffix': 'lidar.npy'
            },
            'sentinel1': {
                'dir': 'Sentinel1_',
                'image_suffix': 'S1.npy'
            },
            'sentinel2': {
                'dir': 'Sentinel2_',
                'image_suffix': 'S2.npy'
            },
            'mask_building': {
                'dir': 'masks_',
                'image_suffix': 'mask_building.npy'
            },
            'mask_platform': {
                'dir': 'masks_',
                'image_suffix': 'mask_platform.npy'
            },
            'mask_aguada': {
                'dir': 'masks_',
                'image_suffix': 'mask_aguada.npy'
            }
        }

        self.input_sources = [x for x in self.data_info.keys() if 'mask' not in x]
        self.label_sources = [x for x in self.data_info.keys() if 'mask' in x]
        self.split_samples = self.get_samples()

        # Get item count also checks if all the sources are consistent so we use it instead of len(self.split_samples)
        if split == 'test':
            self.item_count = self.get_item_count(self.input_sources)
        else:
            self.item_count = self.get_item_count(list(self.data_info.keys()))

        # Aguada is only present in about 3% of the samples, therefore we over sample images with Aguada (factor of 10)
        if self.split != 'test':
            self.sampling_weights = self.weight_resampler()

        logging.info(f'Creating dataset with {self.item_count} examples')
        logging.info(f'Creating dataset with {self.item_count} examples')

    def get_samples(self):
        lidar_dir = self.data_info['lidar']['dir']
        lidar_path = os.path.join(self.imgs_dir_root, f"{lidar_dir}{self.split}")
        samples = [int(f.split('_')[1]) for f in sorted(os.listdir(lidar_path))]
        return samples

    def get_item_count(self, k):
        first_dir = self.data_info[k[0]]['dir']
        first_path = os.path.join(self.imgs_dir_root, f"{first_dir}{self.split}")
        count = len(os.listdir(first_path))
        for x in k[1:]:
            curr = len(os.listdir(os.path.join(self.imgs_dir_root, f"{self.data_info[x]['dir']}{self.split}")))
            if count != curr:
                ValueError(f'Unequal number of files {x}, {curr}, {count}')
        return count

    def weight_resampler(self, aguada_factor=10):
        sw = [1] * self.item_count
        for i in range(self.item_count):
            s = self.__getitem__(i, apply_transform=False)
            # Before normalization the no mask/aguada == 255, therefore if mean == 255 means that there is no aguada in the image
            if s['mask_aguada'].mean() != 255:  # i.e image contains aquada
                sw[i] = sw[i] * aguada_factor
        return sw

    def __len__(self):
        return self.item_count

    def get_id_path(self, index, source_type):
        if 'mask' in source_type and self.split == 'test':
            raise ValueError("Let's not get oversmart ;)")
        di = self.data_info[source_type]
        return os.path.join(self.imgs_dir_root, di['dir'] + self.split, f"tile_{index}_{di['image_suffix']}")

    def __getitem__(self, index, apply_transform=True):
        p_key = self.split_samples[index]
        sample = {
            'idx': p_key,
        }

        for source_type in self.data_info.keys():
            # we don't have masks for test data
            if not (self.split == 'test' and 'mask' in source_type):
                id_path = self.get_id_path(p_key, source_type)
                ds = np.load(id_path)
                sample[source_type] = ds
            else:
                sample[source_type] = np.array([])
        lidar_data = sample['lidar']

        for m in self.label_sources:
            if self.split == 'test' and 'mask' in m:
                continue
            assert lidar_data.shape[1:] == sample[m].shape[1:], \
                f'Image and mask {p_key} should be the same size, but are {lidar_data.shape} and {sample[m].shape}'

        if self.transform and apply_transform:
            sample = self.transform(sample)

        if self.clean_masks and apply_transform:
            for m in ['mask_building', 'mask_platform', 'mask_aguada']:
                # print(type(sample[m]))
                if isinstance(sample[m], np.ndarray):
                    msk = sample[m][0]
                elif isinstance(sample[m], torch.Tensor):
                    msk = sample[m][0].numpy()
                else:
                    raise ValueError(f'Unkown datatype: {type(sample[m])}')

                ob = m.replace('mask_', '')
                cm = mask_cleanup(msk, MIN_OBJ_SIZE[ob], MIN_OBJ_EDGE_ANY[ob], MIN_OBJ_EDGE_ALL[ob], MIN_TOTAL[ob])
                cm = np.expand_dims(cm, axis=0)
                sample[m] = torch.FloatTensor(cm)
        return sample


def split_data(val_split, train_set, val_set, n_iter_epoch, batch_size, batch_size_val, num_workers,
               oversampling=True):
    if val_split is None or val_split == 1:
        n_train = n_iter_epoch * batch_size if n_iter_epoch != -1 else len(train_set)
        if oversampling:
            sampling_weights = train_set.sampling_weights
        else:
            sampling_weights = np.ones_like(train_set.sampling_weights)

        train_sampler = WeightedRandomSampler(sampling_weights, n_train, replacement=oversampling)

        train_loader = torch.utils.data.DataLoader(
            train_set, batch_size=batch_size, pin_memory=True,
            num_workers=num_workers, drop_last=True, sampler=train_sampler
        )
        val_loader = torch.utils.data.DataLoader(
            val_set, batch_size=batch_size_val, pin_memory=True, shuffle=False,
            num_workers=num_workers
        )
    else:
        num_train = len(train_set)
        indices = np.arange(num_train)
        rs = np.random.RandomState(42)
        rs.shuffle(indices)
        split = int(np.floor(val_split * num_train))
        n_train = n_iter_epoch * batch_size if n_iter_epoch != -1 else split
        train_idx, val_idx = indices[:split], indices[split:]
        train_subset = Subset(train_set, train_idx)
        val_subset = Subset(val_set, val_idx)

        if oversampling:
            sampling_weights = np.array(train_set.sampling_weights)
        else:
            sampling_weights = np.ones_like(train_set.sampling_weights)
        sampling_weights = sampling_weights[train_idx]

        train_sampler = WeightedRandomSampler(sampling_weights, n_train, replacement=oversampling)
        train_loader = torch.utils.data.DataLoader(
            train_subset, batch_size=batch_size, sampler=train_sampler, drop_last=True,
            num_workers=num_workers, pin_memory=True,
        )
        val_loader = torch.utils.data.DataLoader(
            val_subset, batch_size=batch_size_val, shuffle=False,
            num_workers=num_workers, pin_memory=True,
        )
    return train_loader, val_loader
