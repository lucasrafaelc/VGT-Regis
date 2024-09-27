from tqdm import tqdm
import glob
import os.path
import cv2
import pandas as pd
import numpy as np
import cv2 as cv


if __name__ == '__main__':
    df = pd.read_excel('dataset/dataset.xlsx', )
    z = 4
    indices_to_keep = np.array([True for i in range(df.shape[0])])
    os.makedirs(f'dataset/out-MAD-{z}', exist_ok=True)
    inputs = glob.glob('result/AAPG-ALL/*')

    for pdf in tqdm(inputs, total=len(inputs)):
        heights = []
        widths = []
        indices = []

        # median and MAD are calculate for each PDF
        for i, row in df.iterrows():
            img = cv2.imread(row['image'])
            if os.path.basename(pdf) in row['image']:
                h, w = img.shape[:2]
                heights.append(h)
                widths.append(w)
                indices.append(i)

        max_w = np.median(widths) + np.median([np.absolute(wi - np.median(widths)) for wi in widths]) * 1.4826 * z
        min_w = np.median(widths) - np.median([np.absolute(wi - np.median(widths)) for wi in widths]) * 1.4826 * z

        min_h = np.median(heights) - np.median([np.absolute(hi - np.median(heights)) for hi in heights]) * 1.4826 * z
        max_h = np.median(heights) + np.median([np.absolute(hi - np.median(heights)) for hi in heights]) * 1.4826 * z

        out = 0

        for i, h, w in zip(indices, heights, widths):
            name = os.path.basename(df.iloc[i]['image'])
            img = cv.imread(f'dataset/pairs/{name}')

            if not (min_w < w < max_w and min_h < h < max_h):
                cv.imwrite(f'dataset/out-MAD-{z}/{name}', img)
                out += 1
                indices_to_keep[i] = False

        # print(os.path.basename(pdf))
        # print(f'keep: {len(indices) - out}, remove: {out}')
        # print(f'width threshold: {(max_w - min_w)/2}, height threshold: {(max_h - min_h)/2}')

    # remove outliers
    df = df.iloc[indices_to_keep]
    print(f'total removed pairs: {len(indices_to_keep) - df.shape[0]}')
    df.to_excel('dataset/dataset_MAD.xlsx')
