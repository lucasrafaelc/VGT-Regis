import os
import cv2 as cv
import numpy as np
import pandas as pd


def pair_visualization(fig_path, text_path, save_path):
    fig = cv.imread(fig_path)
    txt = cv.imread(text_path)
    if fig.shape[0] > txt.shape[0]:
        new_img = np.ones((fig.shape[0], txt.shape[1], 3), dtype=np.uint8)
        new_img = new_img * 255
        new_img[:txt.shape[0], :txt.shape[1], :] = txt
        cv.imwrite(save_path, cv.hconcat([fig, new_img]))

    else:
        new_img = np.ones((txt.shape[0], fig.shape[1], 3), dtype=np.uint8)
        new_img = new_img * 255
        new_img[:fig.shape[0], :fig.shape[1], :] = fig
        cv.imwrite(save_path, cv.hconcat([new_img, txt]))


if __name__ == '__main__':
    df = pd.read_excel('dataset-5e/dataset.xlsx')
    os.makedirs('dataset-5e/pairs/', exist_ok=True)
    for i, row in df.iterrows():
        # print(row[1], row[2])
        origin = os.path.basename(row['image'])
        pair_visualization(row['image'], row['text'], f'dataset-5e/pairs/{origin}')

