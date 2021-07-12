
import torch
import logging


def tb_log_loss(writer, split, loss, global_step):
    writer.add_scalar(
        f'Loss/{split}', ((loss['building'] + loss['platform'] + loss['aguada']) / 3), global_step)
    writer.add_scalar(f'Loss/{split}_building', loss['building'], global_step)
    writer.add_scalar(f'Loss/{split}_platform', loss['platform'], global_step)
    writer.add_scalar(f'Loss/{split}_aguada', loss['aguada'], global_step)


def tb_log_iou(writer, split, iou, global_step):
    writer.add_scalar(
        f'IOU/{split}', (iou['building'] + iou['platform'] + iou['aguada']) / 3, global_step)
    writer.add_scalar(f'IOU-Building/{split}', iou['building'], global_step)
    writer.add_scalar(f'IOU-Platform/{split}', iou['platform'], global_step)
    writer.add_scalar(f'IOU-Aguada/{split}', iou['aguada'], global_step)


def tb_log_images(writer, split, lidar, sentinel2, mask_building, mask_platform, mask_aguada, masks_pred, global_step):
    writer.add_images(f'images/{split}/lidar', lidar, global_step)
    writer.add_images(f'images/{split}/sentinel2_rgb', sentinel2[:, [0, 1, 2]], global_step)
    writer.add_images(f'images/{split}/sentinel2_nir', sentinel2[:, [3]], global_step)

    writer.add_images(f'Building/gt_{split}', mask_building, global_step)
    writer.add_images(
        f'Building/pred_{split}', torch.sigmoid(masks_pred[:, [0]]), global_step)
    writer.add_images(f'Platform/gt_{split}', mask_platform, global_step)
    writer.add_images(
        f'Platform/pred_{split}', torch.sigmoid(masks_pred[:, [1]]), global_step)
    writer.add_images(f'Aguada/gt_{split}', mask_aguada, global_step)
    writer.add_images(
        f'Aguada/pred_{split}', torch.sigmoid(masks_pred[:, [2]]), global_step)


def print_log_loss(writer, split, loss, global_step):

    loss_total = (loss['building'] + loss['platform'] + loss['aguada']) / 3
    print(f'{split} Loss: {loss_total}')
    print(f'{split} Loss per class: {loss}')
    logging.info(f'{split} Loss: {loss_total}')
    logging.info(f'{split} Loss per class: {loss}')
    tb_log_loss(writer, "train", loss, global_step)


def print_log_iou(writer, split, iou, global_step):

    iou_total = (iou['building'] + iou['platform'] + iou['aguada']) / 3
    print(f'{split} IOU: {iou_total}')
    print(f'{split} IOU per class: {iou}')
    logging.info(f'{split} IOU: {iou_total}')
    logging.info(f'{split} IOU per class: {iou}')
    tb_log_iou(writer, "train", iou, global_step)

