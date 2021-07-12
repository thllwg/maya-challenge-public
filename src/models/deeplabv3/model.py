import torch
from torch import nn as nn
from torch.nn import functional as F
from torchvision.models.segmentation.deeplabv3 import DeepLabHead, ASPP
from torchvision import models

def download(url, filename):

    import functools
    import pathlib
    import shutil
    import requests
    from tqdm.auto import tqdm

    r = requests.get(url, stream=True, allow_redirects=True)
    if r.status_code != 200:
        r.raise_for_status()  # Will only raise for 4xx codes, so...
        raise RuntimeError(f"Request to {url} returned status code {r.status_code}")
    file_size = int(r.headers.get('Content-Length', 0))

    path = pathlib.Path(filename).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    desc = "(Unknown total file size)" if file_size == 0 else ""
    r.raw.read = functools.partial(r.raw.read, decode_content=True)  # Decompress if needed
    with tqdm.wrapattr(r.raw, "read", total=file_size, desc=desc) as r_raw:
        with path.open("wb") as f:
            shutil.copyfileobj(r_raw, f)

    return path
    
class MyDeepLabHead(nn.Sequential):

    def __init__(self, in_channels, num_classes, p=0.3, n_feature_maps=512):
        super(MyDeepLabHead, self).__init__(
            ASPP(in_channels, [12, 24, 36]),
            nn.Conv2d(256, n_feature_maps, 3, padding=1, bias=False),
            nn.BatchNorm2d(n_feature_maps),
            nn.ReLU(),
            nn.Dropout(p=p),
            nn.Conv2d(n_feature_maps, num_classes, 1)
        )

class MyDeepLabHeadV1(nn.Sequential):

    def __init__(self, in_channels, num_classes):
        super(MyDeepLabHeadV1, self).__init__(
            ASPP(in_channels, [12, 24, 36]),
            nn.Conv2d(256, 512, 3, padding=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(512, 256, 3, padding=1, bias=False),            
            nn.ReLU(),
            nn.Dropout(p=0.3),            
            nn.Conv2d(256, num_classes, 1)
        )
        
class MyDeepLabHeadV2(nn.Sequential):

    def __init__(self, in_channels, num_classes):
        super(MyDeepLabHeadV2, self).__init__(
            ASPP(in_channels, [12, 24, 36]),
            nn.Conv2d(256, 56, 3, padding=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(56, num_classes, 1)
        )
        
class MyDeepLabHeadV3(nn.Sequential):

    def __init__(self, in_channels, num_classes):
        super(MyDeepLabHeadV3, self).__init__(
            ASPP(in_channels, [12, 24, 36]),
            nn.Conv2d(256, 512, 5, padding=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(512, 256, 5, padding=1, bias=False),            
            nn.ReLU(),
            nn.Dropout(p=0.2),            
            nn.Conv2d(256, num_classes, 1)
        )
        
class MyDeepLabHeadV4(nn.Sequential):

    def __init__(self, in_channels, num_classes):
        super(MyDeepLabHeadV4, self).__init__(
            ASPP(in_channels, [12, 24, 36]),
            nn.Conv2d(256, 1024, 5, padding=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(1024, 1024, 5, padding=1, bias=False),            
            nn.ReLU(),
            nn.Dropout(p=0.5),            
            nn.Conv2d(1024, num_classes, 1)
        )
                                
class DeepLabV3(nn.Module):

    MODEL_URL = "https://uni-muenster.sciebo.de/s/k6xIPWzJ50mm9nB/download?path=%2Fbce_0.0003_0_no-oversampling_bs-16_Jun26_22-31-51_4_eb8a7a46f1b0&files=BestModel_406.pth"
    #MODEL_URL = "https://uni-muenster.sciebo.de/remote.php/webdav/torch/maya/deeplabv3_bce_0.0003_bs-16_no-oversampling_sampling-pixelshuffle_nup-none_no-selfatt_Jun27_15-17-07_7_eb8a7a46f1b0/BestModel.pth"
    
    def __init__(
            self,
            num_classes: int,
            pretrained: bool = True,
            upsampling: str = 'nearest',
            upsampling_sentinel2: str = "bilinear",
            dropout_rate: float = 0.3,
            n_feature_maps: int = 256,
            n_freeze_backbone_epochs: int = 10,
            version: str = "v0",
            device = None
    ):

        super().__init__()

        self.num_classes = num_classes
        self.pretrained = pretrained
        self.upsampling = upsampling
        self.upsampling_sentinel2 = upsampling_sentinel2
        self.dropout_rate = dropout_rate
        self.n_feature_maps = n_feature_maps
        self.n_freeze_backbone_epochs = n_freeze_backbone_epochs
        self.version = version
        self.device = device
        
        # keep track of current epoch
        self._current_epoch = 0

        # PRETRAINED LIDAR/SENTINEL BACKBONE
        # pretrained baseline model
        self.model = models.segmentation.deeplabv3_resnet101(pretrained=False, progress=False)
        self.model.backbone.conv1 = torch.nn.Conv2d(7, 64, (7, 7), (2, 2), (3, 3), bias=False)
        self.model.classifier = MyDeepLabHead(2048, self.num_classes, p=0.1, n_feature_maps=512)
        
        if self.pretrained:
        
            # load pretrained weights from MODEL_URL
            try:
                st_dict = torch.load("basemodel.pth", map_location=self.device)
            except FileNotFoundError:
                print("pre trained model file not found, downloading file and trying one more time..")
                download(self.MODEL_URL, "basemodel.pth")
                st_dict = torch.load("basemodel.pth", map_location=self.device)
            self.load_state_dict(st_dict['net_params'])
        
        # freeze all backbone layers
        for param in self.model.parameters():
            param.requires_grad = False        
            
        # define new head based on version
        if self.version == "v0":
        
            self.model.classifier = MyDeepLabHead(2048, self.num_classes, p=self.dropout_rate, n_feature_maps=self.n_feature_maps)
            
        elif self.version == "v1":

            self.model.classifier = MyDeepLabHeadV1(2048, self.num_classes)

        elif self.version == "v2":

            self.model.classifier = MyDeepLabHeadV2(2048, self.num_classes)
            
        elif self.version == "v3":

            self.model.classifier = MyDeepLabHeadV3(2048, self.num_classes)
            
        elif self.version == "v4":

            self.model.classifier = MyDeepLabHeadV4(2048, self.num_classes)
                                                
        else:
        
            raise Exception(f'Unkown model version {self.version}')
            
        # upsampling
        self.sentinel_upsample = lambda x, size: F.interpolate(
            x, size=size, mode=self.upsampling_sentinel2, align_corners=True
        )

    def forward(self, lidar, sentinel2):

        sentinel2 = self.sentinel_upsample(sentinel2, lidar.shape[2:])
        x = torch.cat([lidar, sentinel2], dim=1)

        return self.model(x)['out']
        
    def set_current_epoch(self, current_epoch):
    
        self._current_epoch = current_epoch

        if self.training and (self._current_epoch == self.n_freeze_backbone_epochs):

            # unfreeze all backbone layers
            for param in self.model.parameters():
                param.requires_grad = True

    @classmethod
    def load_model(nn_model_cls, path, device):

        st_dict = torch.load(path, map_location=device)

        net = nn_model_cls(num_classes=st_dict['num_classes'],
                        pretrained=st_dict['pretrained'],
                        upsampling=st_dict['upsampling'],
                        dropout_rate=st_dict['dropout_rate'],
                        n_feature_maps=st_dict['n_feature_maps'],
                        n_freeze_backbone_epochs=st_dict['n_freeze_backbone_epochs'],   
                        version=st_dict['version'],                           
                        device=device,                                             
                )
        net.load_state_dict(st_dict['net_params'])
        net.to(device=device)
        return net

    @staticmethod
    def load_optimizer(path, optm, device):

        st_dict = torch.load(path, map_location=device)
        optm.load_state_dict(st_dict['optm_params'])

        return optm

    @staticmethod
    def save_model(model, path, optm=None):

        net = model

        if isinstance(model, nn.DataParallel):
            net = model.module

        st_dict = {'net_params': net.state_dict()}

        if optm is not None:
            st_dict['optm_params'] = optm.state_dict()

        st_dict['num_classes'] = net.num_classes
        st_dict['pretrained'] = net.pretrained
        st_dict['upsampling'] = net.upsampling
        st_dict['upsampling_sentinel2'] = net.upsampling_sentinel2
        st_dict['dropout_rate'] = net.dropout_rate
        st_dict['n_feature_maps'] = net.n_feature_maps
        st_dict['n_freeze_backbone_epochs'] = net.n_freeze_backbone_epochs                        
        st_dict['version'] = net.version

        try:
            torch.save(st_dict, path)
            return 1
        except Exception as e:
            return 0


