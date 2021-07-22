import argparse
import logging
import os
import random
import shutil
import socket
import sys
from datetime import datetime

import torch
import torch.nn as nn
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.data.maya_dataset import MayaDataset, MayaTransform, split_data
from src.models import str2bool
from src.models.deeplabv3 import DeepLabV3
from src.models.unet_adv import UNet as UNetA
from src.models.eval_model import eval_net
from src.models.losses import maya_iou, FocalTverskyLoss, add_to_dataset_loss, normalize_loss, add_values, div_values, \
    VGGLoss
from src.utils.board_logger import print_log_iou, print_log_loss, tb_log_images


def train_net(net,
              train_loader,
              val_loader,
              optimizer,
              epochs=5,
              batch_size=1,
              save_cp=True,
              img_type=torch.float32,
              mask_type=torch.float32,
              loss_function="bce",
              focal_gamma=1,
              loss_alpha=.5,
              loss_beta=.5,
              run_dir='../runs/',
              iou_absent=1,
              n_grad_accumulation=1,
              model_type="unet",
              verbose_logging=True,
              checkpoint_per_epoch=True,
              lr_scheduler=False,
              lr_scheduler_warmup=0,
              exclude_aguada=False):
    n_train = (len(train_loader) // n_grad_accumulation) * n_grad_accumulation
    n_val = len(val_loader)

    model_dir = os.path.join(run_dir, 'model')
    tensorboard_dir = os.path.join(run_dir, 'tensorboard')
    writer = SummaryWriter(log_dir=tensorboard_dir)
    global_step = 0

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_cp}
        run_dir:         {run_dir}
    ''')

    if loss_function == "bce":
        criterion = nn.BCEWithLogitsLoss(reduction="mean")
    elif loss_function == "focaltversky":
        criterion = FocalTverskyLoss(alpha=loss_alpha, beta=loss_beta, gamma=focal_gamma, reduction="mean")
    elif loss_function == "vgg":
        criterion = VGGLoss()
        criterion.to(device)
    else:
        raise Exception(f"No loss function called '{loss_function}' available.")

    if lr_scheduler == "cosinewr":
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 10, 2)
    elif lr_scheduler == "none":
        class NoScheduler():
            def step(self, epoch): return

        scheduler = NoScheduler()
    else:
        raise Exception(f"lr scheduler {lr_scheduler} not implemented.")

    best_iou = 0

    for epoch in range(0, epochs):

        net.train()

        if model_type in ["deeplabv3", "unet_adv"]:
            net.set_current_epoch(epoch)

        epoch_loss = {
            'building': 0,
            'platform': 0,
            'aguada': 0
        }
        epoch_iou = {
            'building': 0,
            'platform': 0,
            'aguada': 0
        }

        with tqdm(total=n_train * batch_size, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            iou = {
                "building": 0.,
                "platform": 0.,
                "aguada": 0.,
            }
            loss = {
                "building": 0.,
                "platform": 0.,
                "aguada": 0.,
            }
            for i, batch in enumerate(train_loader):
                lidar = batch['lidar'].to(device=device, dtype=img_type)
                sentinel2 = batch['sentinel2'].to(device=device, dtype=img_type)
                mask_building = batch['mask_building'].to(
                    device=device, dtype=mask_type)
                mask_platform = batch['mask_platform'].to(
                    device=device, dtype=mask_type)
                mask_aguada = batch['mask_aguada'].to(
                    device=device, dtype=mask_type)

                masks_pred = net(lidar, sentinel2)

                b = criterion(masks_pred[:, [0]], mask_building)
                p = criterion(masks_pred[:, [1]], mask_platform)

                if exclude_aguada:
                    with torch.no_grad():
                        a = criterion(masks_pred[:, [2]], mask_aguada)

                    tloss = (b + p) / 2
                else:
                    a = criterion(masks_pred[:, [2]], mask_aguada)
                    tloss = (b + p + a) / 3

                iou = add_values(
                    iou,
                    maya_iou(masks_pred, mask_building, mask_platform, mask_aguada, absent_score=iou_absent)
                )

                loss = add_values(loss, {'building': b.item(), 'platform': p.item(), 'aguada': a.item()})

                # nn.utils.clip_grad_value_(net.parameters(), 0.1) # TODO: Why do we need gradient clipping?
                tloss.backward()
                if (i + 1) % n_grad_accumulation == 0:
                    optimizer.step()
                    optimizer.zero_grad()

                    # div by n_grad_accumulation
                    loss = div_values(loss, n_grad_accumulation)
                    iou = div_values(iou, n_grad_accumulation)

                    epoch_loss = add_to_dataset_loss(epoch_loss, loss)
                    epoch_iou = add_to_dataset_loss(epoch_iou, iou)

                    pbar.set_postfix(**{'loss': ((loss['building'] + loss['platform'] + loss['aguada']) / 3),
                                        'loss building': loss['building'], 'loss platform': loss['platform'],
                                        'loss aguada': loss['aguada'],
                                        'iou': (iou['building'] + iou['platform'] + iou['aguada']) / 3,
                                        'iou building': iou['building'], 'iou platform': iou['platform'],
                                        'iou aguada': iou['aguada']})

                    pbar.update(lidar.shape[0] * n_grad_accumulation)
                    global_step += 1

                    iou = {
                        "building": 0.,
                        "platform": 0.,
                        "aguada": 0.,
                    }
                    loss = {
                        "building": 0.,
                        "platform": 0.,
                        "aguada": 0.,
                    }

                    if epoch + 1 > lr_scheduler_warmup:
                        scheduler.step(epoch - lr_scheduler_warmup + i / n_train)

                    if i + 1 == n_train:
                        break

            epoch_loss = normalize_loss(epoch_loss, n_train)
            epoch_iou = normalize_loss(epoch_iou, n_train)
            print_log_loss(writer, "Epoch", epoch_loss, global_step)
            print_log_iou(writer, "Epoch", epoch_iou, global_step)

            if save_cp:
                try:
                    os.mkdir(model_dir)
                    logging.info('Created checkpoint directory')
                except OSError:
                    pass
            if ((epoch + 1) % checkpoint_per_epoch) == 0:
                net.save_model(net, os.path.join(model_dir, f'CP_epoch{epoch + 1}.pth'), optm=optimizer)
                logging.info(f'Checkpoint {epoch + 1} saved !')

            if model_type == "unet" and verbose_logging:
                for tag, value in net.named_parameters():
                    tag = tag.replace('.', '/')
                    writer.add_histogram(
                        'weights/' + tag, value.data.cpu().numpy(), global_step)
                    writer.add_histogram(
                        'grads/' + tag, value.grad.data.cpu().numpy(), global_step)
            writer.add_scalar(
                'learning_rate', optimizer.param_groups[0]['lr'], global_step)

            if verbose_logging:
                tb_log_images(writer, "train", lidar, sentinel2, mask_building, mask_platform, mask_aguada, masks_pred,
                              global_step)

            # Validation at the end of epoch
            val_loss, val_iou = eval_net(net, val_loader, device, criterion, writer, global_step, iou_absent)

            if exclude_aguada:
                val_iou_total = (val_iou['building'] + val_iou['platform']) / 2
            else:
                val_iou_total = (val_iou['building'] + val_iou['platform'] + val_iou['aguada']) / 3

            if best_iou < val_iou_total:

                best_iou = val_iou_total
                best_model_path = os.path.join(model_dir, 'BestModel.pth')

                if not verbose_logging:
                    tb_log_images(writer, "train", lidar, sentinel2, mask_building,
                                  mask_platform, mask_aguada, masks_pred, global_step)

                # keep only the last 4 best-performing models
                try:
                    shutil.move(best_model_path, os.path.join(model_dir, f'BestModel_{epoch:05d}.pth'))

                    model_files = sorted([f for f in os.listdir(model_dir) if
                                          os.path.isfile(os.path.join(model_dir, f)) and f.startswith('BestModel')])
                    for filename in model_files[:-4]:
                        filename_relPath = os.path.join(model_dir, filename)
                        os.remove(filename_relPath)
                except Exception as e:
                    pass

                # save new best model
                net.save_model(net, best_model_path, optm=optimizer)

                writer.add_text("Training/New Best Model",
                                f"New best model saved at epoch {epoch + 1}  \n New Best Validation Loss: {val_loss}  \n New Best Validation IOU {best_iou}",
                                global_step)
                print("     | > Saved as new best model.")
                logging.info(
                    'Saved the best model, found in epoch: {}'.format(epoch + 1))
                logging.info('New Best Validation Loss: {}'.format(val_loss))
                logging.info('New Best Validation IOU: {}'.format(best_iou))

    writer.close()


def get_args():

    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-d', '--dir_img_root', type=str, default='./data/processed/',
                        help='Dir with the images', dest='dir_img_root')
    parser.add_argument('--log-dir', type=str, default='./runs',
                        help='where to save models and tensorboard output')
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=500,
                        help='Number of epochs', dest='epochs')
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, default=8,
                        help='Batch size')
    parser.add_argument('--batch-size-val', metavar='B', type=int, default=None,
                        help='Batch size of validation')
    parser.add_argument('-l', '--learning-rate', metavar='LR', type=float, default=3e-4,
                        help='Learning rate', dest='lr')
    parser.add_argument('--opt-eps', type=float, default=1e-8,
                        help='Epsilon parameter of Adam (default 1e-8)')
    parser.add_argument('-f', '--load', dest='load', type=str, default=False,
                        help='Load model from a .pth file')
    parser.add_argument('--loss-function', type=str, default="bce",
                        help='Which loss function to use (default: "bce", choices: "bce", "focaltversky")')
    parser.add_argument("--focal-gamma", type=float, default=4 / 3, help='Gamma of focal losses (default 4/3)')
    parser.add_argument("--loss-alpha", type=float, default=.7, help='Alpha of balance losses (default .7)')
    parser.add_argument("--loss-beta", type=float, default=.3, help='Beta of focal losses (default .3)')
    parser.add_argument("--val-split", type=float, default=1.0, help='Split ratio for train/val (default .8)')
    parser.add_argument("--iou-absent", type=float, default=1, help='Absent value for IoU calculation (default 1)')
    parser.add_argument('--fusion-idx', type=int, default=0,
                        help='When to merge sentinel 2 images with Lidar images (default: 0)')
    parser.add_argument('-v', '--validation', dest='val_percent', type=float, default=0.1,
                        help='Percent of the data that is used as validation [0.0-1.0]')
    parser.add_argument('--num-workers', type=int, default=0,
                        help='how many data workers per dataloader')
    parser.add_argument('--crop-size', type=int, default=400,
                        help='How much the lidar input is cropped (default: 400)')
    parser.add_argument('--n-iter-epoch', type=int, default=-1, help='how many iterations per epochs')
    parser.add_argument('-md', '--multi-devices', '--names-list', dest='multi_devices',
                        default=None, nargs='+', help='ids of gpus that should be used (e.g., --multi-devices 0 1)')
    parser.add_argument('--n-grad-accumulation', type=int, default=1,
                        help='Number of gradient accumulation step (default 1).')
    parser.add_argument('--advanced-augmentation', action='store_true', default=False)
    parser.add_argument('--oversampling', type=str2bool, nargs='?', default=True,
                        help='Determines if we use oversampling (default True).')
    parser.add_argument('--norm-down', type=str, default="bn",
                        help='Normalization of up path (default: "bn"')
    parser.add_argument('--norm-up', type=str, default="none",
                        help='Normalization of up path, defaults to down normalization.')
    parser.add_argument('--act-fct', type=str, default="elu",
                        help='Activation function to use (default: "elu").')
    parser.add_argument('--upsampling', type=str, default="pixelshuffle",
                        help='Upsampling in decoder, e.g., "pixelshuffle", "nearest", '
                             '"bilinear" (default: "pixelshuffle").')
    parser.add_argument('--arch', type=str, default="mnasnet",
                        help='Architecture of encoder, e.g., "standard", "mnasnet", '
                             '"resnet18", "resnet50", "swin-s", "swin-b" (default: "mnasnet").')
    parser.add_argument('-c', '--clean_masks', type=str2bool, dest='clean_masks', default=False,
                        help='Determines if we clean the masks before training the model (default False).')
    parser.add_argument('--model-type', dest="model_type", type=str, default="unet",
                        help='The model type to be used (if instantiated). Default: "unet"')
    parser.add_argument('--model-version', dest="model_version", type=str, default="v0",
                        help='The model version to be used (if instantiated). Default: "v0"')
    parser.add_argument('--no-self-attention', action='store_true')
    parser.add_argument('--no-blur', action='store_true', default=False)
    parser.add_argument('--verbose-logging', action='store_true', default=False)

    parser.add_argument('--checkpoint-per-epoch', type=int, default=10, help='After how epoch a checkpoint is written')
    parser.add_argument('--lr-scheduler', type=str, default="none",
                        help="Either 'none, 'linear', or 'cosinewr' (default: 'none')")
    parser.add_argument('--lr-scheduler-warmup', type=int, default=0,
                        help="After how many epochs to start using the lr-scheduler (default: 0)")
    parser.add_argument('--backbone-warmup', type=int, default=0,
                        help='Backbone of the warmup network for unet_adv')
    parser.add_argument('--exclude-aguada', action='store_true', default=False)
    return parser.parse_args()


if __name__ == '__main__':

    args = get_args()
    if args.batch_size_val is None:
        args.batch_size_val = args.batch_size

    params = f"{args.model_type}_{args.loss_function}_{args.lr}"
    if args.crop_size != 400:
        params += f"_cs-{args.crop_size}"
    if args.n_grad_accumulation != 1:
        params += f"_nga-{args.n_grad_accumulation}"
    if args.batch_size != 8:
        params += f"_bs-{args.batch_size}"
    if not args.oversampling:
        params += f"_no-oversampling"
        params += f"_sampling-{args.upsampling}"
    if args.loss_function.lower() == "focaltversky":
        params += f"_a-{args.loss_alpha}_b-{args.loss_beta}_g-{args.focal_gamma}"
    if args.lr_scheduler != "none":
        params += f"_lr-{args.lr_scheduler}"
    if args.exclude_aguada:
        params += f"_no-aguada"

    # unet specific
    if args.act_fct != "elu":
        params += f"_act-{args.act_fct}"
    if args.norm_up is not None:
        params += f"_nup-{args.norm_up}"
    if args.norm_down != "bn":
        params += f"_ndown-{args.norm_down}"

    # advanced unet specific
    if args.no_self_attention:
        params += f"_no-selfatt"
    if args.upsampling.lower() != "pixelshuffle":
        params += f"_sampling-{args.upsampling}"
    if args.no_blur:
        params += "_no-blur"
    if args.arch != "mnasnet":
        params += f"_arch-{args.arch}"

    base_run_dir = args.log_dir
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    rs = random.Random(datetime.now())
    hash = rs.getrandbits(4)
    run_dir = os.path.join(base_run_dir, f'{params}_{current_time}_{hash}_{socket.gethostname()}')
    os.makedirs(run_dir, exist_ok=True)
    logs_file = os.path.join(run_dir, 'logs.txt')

    # BasicConfig must be called before any logs are written!
    logging.basicConfig(filename=logs_file, level=logging.INFO, format='%(levelname)s: %(message)s')
    logging.info(f'Writing the run info to {run_dir}')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    if args.model_type == "deeplabv3":
        if args.load:
            net = DeepLabV3.load_model(args.load, device)
            optimizer = optim.Adam(net.parameters(), lr=args.lr, eps=args.opt_eps)
            optimizer = DeepLabV3.load_optimizer(args.load, optimizer, device)
            logging.info(f'Model loaded from {args.load}')
        else:
            net = DeepLabV3(num_classes=3, n_freeze_backbone_epochs=args.backbone_warmup, version=args.model_version)
            net.to(device)
            optimizer = optim.Adam(net.parameters(), lr=args.lr, eps=args.opt_eps)
    elif args.model_type == "unet_adv":
        if args.load:
            net = UNetA.load_model(args.load, device)
            net.backbone_warmup = args.backbone_warmup
            optimizer = optim.AdamW(net.parameters(), lr=args.lr, eps=args.opt_eps,
                                    betas=(0.9, 0.999), weight_decay=0.01)
            optimizer = net.load_optimizer(args.load, optimizer, device)
            logging.info(f'Model loaded from {args.load}')
        else:
            net = UNetA(
                n_classes=3, n_features_lidar=3, n_features_sentinel2=4,
                activation=args.act_fct, norm_type_encoder=args.norm_down,
                norm_type_decoder=args.norm_up, self_attention=not args.no_self_attention,
                arch=args.arch, upsampling=args.upsampling, blur=not args.no_blur, blur_final=not args.no_blur,
                backbone_warmup=args.backbone_warmup
            )
            net.to(device)
            optimizer = optim.AdamW(net.parameters(), lr=args.lr, eps=args.opt_eps,
                                    betas=(0.9, 0.999), weight_decay=0.01)

    else:
        raise Exception(f'Unknowng model type: {args.model_type}')

    logging.info(f'''
        Device:          {device}
        Loss function:   {args.loss_function.lower()}
        Loss alpha:      {args.focal_gamma}
        Loss beta:       {args.loss_beta}
        Focal gamma:     {args.focal_gamma}
        Leaning rate:    {args.lr}
        Steps Grad. Acc.:{args.n_grad_accumulation}
        Oversampling:    {args.oversampling}
        Clean training masks:   {args.clean_masks}
        Exclude Aguada:  {args.exclude_aguada}
        ''')
    if args.model_type == "unet":
        logging.info(f'Network:\n'
                     f'\t{net.input_channels_lidar} input lidar channels\n'
                     f'\t{net.input_channels_sentinel2} input sentinel2 channels (classes)\n'
                     f'\t{net.num_classes} output channels (classes)\n'
                     f'\t{net.upsampling} network upsampling mode\n'
                     f'\t{net.upsampling_sentinel2} sentinel2 upsampling mode\n'
                     f'\t{net.idx_fusion} sentinel2 fusion layer\n'
                     f'\t{args.act_fct} activation function\n'
                     f'\t{args.norm_down} Norm down\n'
                     f'\t{args.norm_up if args.norm_down is not None else args.norm_down} Norm up\n')
    elif args.model_type == "unet_adv":
        logging.info(f'Network:\n'
                     f'\t{net.n_features_lidar} input lidar channels\n'
                     f'\t{net.n_features_sentinel2} input sentinel2 channels (classes)\n'
                     f'\t{net.n_classes} output channels (classes)\n'
                     f'\t{net.upsampling} network upsampling mode\n'
                     f'\t{net.activation} activation function\n'
                     f'\t{args.norm_down} Norm down\n'
                     f'\t{args.norm_up if args.norm_down is not None else args.norm_down} Norm up\n'
                     f'\tBlur {net.blur}\n'
                     f'\tSelf-attention {not net.self_attention}\n')
    elif args.model_type == "deeplabv3":
        logging.info(f'Network:\n'
                     f'\t{net.num_classes} output channels (classes)\n'
                     f'\t{net.version} network version\n'
                     f'\t{net.n_freeze_backbone_epochs} number of epochs for backbone freeze\n')

if args.multi_devices is not None:
    try:
        md = [int(i) for i in args.multi_devices]
        net_multi = nn.DataParallel(net, device_ids=md)
        net = net_multi
        logging.info(f'Using multiple GPU devices: {md}')
    except Exception:
        logging.error(
            f'Could not set multiple GPU devices ({args.multi_devices}). Using only single device ...')

train_dataset = MayaDataset(
    args.dir_img_root, split="train", transform=MayaTransform(
        use_augmentations=True, use_advanced_augmentations=args.advanced_augmentation, crop_size=args.crop_size
    ), clean_masks=args.clean_masks
)
val_dataset = MayaDataset(
    args.dir_img_root, split="train", transform=MayaTransform(use_augmentations=False))

train_loader, val_loader = split_data(
    args.val_split, train_dataset, val_dataset,
    args.n_iter_epoch, args.batch_size, args.batch_size_val,
    args.num_workers, args.oversampling
)

try:
    train_net(net=net,
              train_loader=train_loader,
              val_loader=val_loader,
              optimizer=optimizer,
              epochs=args.epochs,
              batch_size=args.batch_size,
              loss_function=args.loss_function.lower(),
              focal_gamma=args.focal_gamma,
              loss_alpha=args.loss_alpha,
              loss_beta=args.loss_beta,
              run_dir=run_dir,
              iou_absent=args.iou_absent,
              n_grad_accumulation=args.n_grad_accumulation,
              model_type=args.model_type,
              verbose_logging=args.verbose_logging,
              checkpoint_per_epoch=args.checkpoint_per_epoch,
              lr_scheduler=args.lr_scheduler,
              lr_scheduler_warmup=args.lr_scheduler_warmup,
              exclude_aguada=args.exclude_aguada
              )
except KeyboardInterrupt:
    torch.save(net.state_dict(), 'INTERRUPTED.pth')
    logging.info('Saved interrupt')
    try:
        sys.exit(0)
    except SystemExit:
        os._exit(0)
