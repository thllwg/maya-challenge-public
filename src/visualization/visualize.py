import numpy as np
import matplotlib.pyplot as plt

def visualize_output_of_dataloader(dataloader, rows = 10, cols = 16 ):
    images_so_far = 0
    fig = plt.figure(figsize=(50, 50 * rows // cols))
    for i_batch, sample_batched in enumerate(dataloader):
        sample_keys = [k for k in list(sample_batched.keys()) if 'sentinel1' not in k and 'idx' not in k]
        for j in range(sample_batched[sample_keys[0]].size()[0]):
            for k in sample_keys:
                if 'lidar' in k:
                    images_so_far += 1
                    ax = plt.subplot(rows, cols, images_so_far)
                    ax.axis('off')
                    dt = sample_batched[k]
                    img = dt.cpu().data[j].numpy()[[0]]
                    ax.set_title(f'{k}: mean: {img.mean():.2f},\n min: {img.min():.2f}, max: {img.max():.2f}')
                    plt.imshow(img.transpose(1,2,0))

                    images_so_far += 1
                    ax = plt.subplot(rows, cols, images_so_far)
                    ax.axis('off')
                    dt = sample_batched[k]
                    img = dt.cpu().data[j].numpy()[[1]]
                    ax.set_title(f'{k}: mean: {img.mean():.2f},\n min: {img.min():.2f}, max: {img.max():.2f}')
                    plt.imshow(img.transpose(1,2,0))

                    images_so_far += 1
                    ax = plt.subplot(rows, cols, images_so_far)
                    ax.axis('off')
                    dt = sample_batched[k]
                    img = dt.cpu().data[j].numpy()[[2]]
                    ax.set_title(f'{k}: mean: {img.mean():.2f},\n min: {img.min():.2f}, max: {img.max():.2f}')
                    plt.imshow(img.transpose(1,2,0))

                elif 'sentinel2' in k:
                    # RGB
                    images_so_far += 1
                    ax = plt.subplot(rows, cols, images_so_far)
                    ax.axis('off')
                    dt = sample_batched[k]
                    img = dt.cpu().data[j].numpy()[:3]
                    ax.set_title(f'{k}: mean: {img.mean():.2f},\n min: {img.min():.2f}, max: {img.max():.2f}')
                    plt.imshow(img.transpose(1,2,0))

                    # NiR
                    images_so_far += 1
                    ax = plt.subplot(rows, cols, images_so_far)
                    ax.axis('off')
                    dt = sample_batched[k]
                    img = dt.cpu().data[j].numpy()[[3]]
                    ax.set_title(f'{k}: mean: {img.mean():.2f},\n min: {img.min():.2f}, max: {img.max():.2f}')
                    plt.imshow(img.transpose(1,2,0))
                else: 
                    images_so_far += 1
                    ax = plt.subplot(rows, cols, images_so_far)
                    ax.axis('off')
                    dt = sample_batched[k]
                    img = dt.cpu().data[j].numpy()[:3]
                    ax.set_title(f'{k}: mean: {img.mean():.2f},\n min: {img.min():.2f}, max: {img.max():.2f}')
                    plt.imshow(img.transpose(1,2,0))

                if images_so_far == rows * cols:
                    return
