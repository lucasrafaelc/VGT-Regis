import os

import numpy as np
from tqdm import tqdm
import clip
import torch
import pandas as pd
from PIL import Image

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load('ViT-L/14', device)

    classes = ['chart', 'petrography', 'diagram', 'person']
    texts = torch.cat([clip.tokenize(f'a photo of a {label}') for label in classes]).to(device)

    os.makedirs('dataset/out_clip_filter', exist_ok=True)
    df = pd.read_excel('dataset/dataset_MAD.xlsx')

    indices_to_keep = np.array([True for i in range(df.shape[0])])

    with torch.no_grad():
        texts_features = model.encode_text(texts)
        texts_features /= texts_features.norm(dim=-1, keepdim=True)

        counter = 0
        for i, row in tqdm(df.iterrows(), total=df.shape[0]):
            image_path = row['image']
            image = Image.open(image_path)
            image_features = model.encode_image(preprocess(image).unsqueeze_(0).to(device))
            image_features /= image_features.norm(dim=-1, keepdim=True)
            similarity = (100. * image_features @ texts_features.T).softmax(dim=-1)
            value, index = similarity[0].topk(1, dim=-1)

            if index[0] != 1:
                pair = Image.open('dataset/pairs/' + os.path.basename(image_path))
                pair.save('dataset/out_clip_filter/' + os.path.basename(image_path))
                indices_to_keep[i] = False
                counter += 1

        df = df.iloc[indices_to_keep]
        df.to_excel('dataset/dataset_clip_filter.xlsx')
        print(f'total removed pairs: {counter}')

