from src.utils.board_logger import print_log_iou, print_log_loss, tb_log_images, tb_log_iou, tb_log_loss
import torch
from tqdm.auto import tqdm
from src.models.losses import maya_dice_coeff, maya_iou, add_to_dataset_loss, normalize_loss
import logging

from src.models.losses import maya_dice_coeff, maya_iou


def eval_net(net,
            loader,
            device,
            loss_criterion,
            writer,
            logging,
            global_step,
            iou_absent=1.0,
            img_type = torch.float32,
            mask_type = torch.float32
        ):
    """Evaluation without the densecrf with the dice coefficient"""
    net.eval()
    n_val = len(loader)  # the number of batch

    val_loss = {
        'building': 0,
        'platform': 0,
        'aguada': 0
    }

    val_iou = {
        'building': 0,
        'platform': 0,
        'aguada': 0
    }

    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        for batch in loader:
            lidar = batch['lidar'].to(device=device, dtype=img_type)
            sentinel2 = batch['sentinel2'].to(device=device, dtype=img_type)
            mask_building = batch['mask_building'].to(device=device, dtype=mask_type)
            mask_platform = batch['mask_platform'].to(device=device, dtype=mask_type)
            mask_aguada = batch['mask_aguada'].to(device=device, dtype=mask_type)

            with torch.no_grad():
                masks_pred = net(lidar, sentinel2)

            loss = {
                'building' : loss_criterion(masks_pred[:, [0]], mask_building).item(),
                'platform' : loss_criterion(masks_pred[:, [1]], mask_platform).item(),
                'aguada' : loss_criterion(masks_pred[:, [2]], mask_aguada).item(),
            } 

            iou = maya_iou(masks_pred, mask_building, mask_platform, mask_aguada, absent_score=iou_absent)

            val_loss =  add_to_dataset_loss(val_loss, loss)
            val_iou = add_to_dataset_loss(val_iou, iou)

            pbar.update()

    val_loss = normalize_loss(val_loss, n_val )
    val_iou = normalize_loss(val_iou, n_val)

    print_log_loss(writer, "Validation", val_loss, global_step)
    print_log_iou(writer, "Validation", val_iou, global_step)

    tb_log_loss(writer, "val", val_loss, global_step)
    tb_log_iou(writer, "val", val_iou, global_step)

    tb_log_images(writer, "val", lidar, sentinel2, mask_building, mask_platform, mask_aguada, masks_pred, global_step)

    net.train()
    return val_loss, val_iou
