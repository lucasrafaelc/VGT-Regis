import argparse
import pickle
from labels import labels
import cv2
from detectron2.utils.visualizer import ColorMode, Visualizer
import os
from VGT.object_detection.ditod import add_vit_config
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
import cv2
import glob
from non_max import non_maximum_suppression
from tqdm import tqdm


def save_result(image_path, conf, classes, threshold):
    img = cv2.imread(image_path)
    md = MetadataCatalog.get(conf.DATASETS.TEST[0])
    md.set(thing_classes=classes)
    page = os.path.basename(image_path)
    trailing = os.path.dirname(os.path.dirname(image_path))
    output_path = os.path.join(trailing, 'outputs', page).split('.png')[0] + '.pkl'

    if os.path.exists(output_path):
        output = pickle.load(open(output_path, 'rb'))
        output = non_maximum_suppression(output, threshold)

        v = Visualizer(img[:, :, ::-1],
                       md,
                       scale=1.0,
                       instance_mode=ColorMode.SEGMENTATION)
        result_image = v.draw_instance_predictions(output.to("cpu"))
        result_image = result_image.get_image()[:, :, ::-1]
        result_path = os.path.join(trailing, 'result_visualize', page)

        if not os.path.exists(os.path.dirname(result_path)):
            os.makedirs(os.path.dirname(result_path))
        cv2.imwrite(result_path, result_image)


def extract_bbox(image_path, label, threshold,):
    img = cv2.imread(image_path)
    page = os.path.basename(image_path)
    trailing = os.path.dirname(os.path.dirname(image_path))
    output_path = os.path.join(trailing, 'outputs', page).split('.')[0] + '.pkl'
    output = pickle.load(open(output_path, 'rb'))
    output = non_maximum_suppression(output, threshold)

    for i in range(len(output)):
        # print(output[i])
        if output[i].pred_classes.item() == label:
            x1 = int(output[i].pred_boxes.tensor.squeeze()[0].item())
            y1 = int(output[i].pred_boxes.tensor.squeeze()[1].item())
            x2 = int(output[i].pred_boxes.tensor.squeeze()[2].item())
            y2 = int(output[i].pred_boxes.tensor.squeeze()[3].item())
            figure = img[y1:y2, x1:x2, :]
            page_number = page.split('.')[0]
            figure_path = os.path.join(trailing, f'cropped_{interesting_label}', f'{page_number}_box{i}.png')
            if not os.path.exists(os.path.dirname(figure_path)):
                os.makedirs(os.path.dirname(figure_path))

            cv2.imwrite(figure_path, figure)


if __name__ == '__main__':
    cfg = get_cfg()
    add_vit_config(cfg)
    cfg.merge_from_file('configs/cascade/doclaynet_VGT_cascade_PTM.yaml')
    files = glob.glob('result/teste/20725-1225-16699-1-10-20220719/pages/*')
    dataset = 'doclaynet'
    interesting_label = ''

    img_classes = {
        'doclaynet': 'Picture',
        'publaynet': 'figure',
        'D4LA': 'Figure',
        'dockbank': 'figure'
    }
    txt_classes = {
        'doclaynet': 'Text',
        'publaynet': 'text'
    }

    print('processing...')
    for path in tqdm(files, total=len(files)):
        save_result(path, cfg, labels[dataset], 0.2)
    print('Done!')



