import cv2
from nms import nms
import pickle
from detectron2.structures.instances import Instances
from detectron2.structures.boxes import Boxes
import torch
from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2.data import MetadataCatalog
from VGT.object_detection.ditod import add_vit_config
from detectron2.config import get_cfg
from labels import labels


def non_maximum_suppression(output, iou_threshold=0.3):
    boxes = output.get('pred_boxes').to('cpu'),
    scores = output.get('scores').to('cpu').numpy(),
    classes = output.get('pred_classes').to('cpu').numpy()

    scores_nms = []
    boxes_nms = []

    for i in range(len(classes)):
        x = boxes[0][i].tensor.squeeze()[0].item()
        y = boxes[0][i].tensor.squeeze()[1].item()
        w = boxes[0][i].tensor.squeeze()[2].item() - x
        h = boxes[0][i].tensor.squeeze()[3].item() - y

        boxes_nms.append((x, y, w, h))
        scores_nms.append(scores[0][i])

    indices = nms.boxes(boxes_nms, scores_nms, nms_threshold=iou_threshold)

    selected_boxes = [
        [boxes_nms[i][0], boxes_nms[i][1], boxes_nms[i][0] + boxes_nms[i][2], boxes_nms[i][1] + boxes_nms[i][3]] for i
        in indices]

    data = {
        'pred_boxes': Boxes(torch.tensor(selected_boxes)).to('cuda'),
        'scores': torch.tensor([scores_nms[i] for i in indices]).to('cuda'),
        'pred_classes': torch.tensor([classes[i] for i in indices]).to('cuda'),
    }
    return Instances(output.image_size, **data), data['scores']

if __name__ == '__main__':
    output_path = 'outputs/pdfs/multfig/AAPG Memoir 77_Colour Guide to the Petrography of Carbonate Rocks_Schole & Schole_2003/page_15.pkl'
    output = non_maximum_suppression(pickle.load(open(output_path, 'rb')))
    img = cv2.imread('images/pdfs/multfig/AAPG Memoir 77_Colour Guide to the Petrography of Carbonate Rocks_Schole & Schole_2003/page_15.png')

    cfg = get_cfg()
    add_vit_config(cfg)
    cfg.merge_from_file('Configs/cascade/doclaynet_VGT_cascade_PTM.yaml')
    md = MetadataCatalog.get(cfg.DATASETS.TEST[0])
    md.set(thing_classes=labels['doclaynet'])

    v = Visualizer(img[:, :, ::-1],
                   md,
                   scale=1.0,
                   instance_mode=ColorMode.SEGMENTATION)

    result_image = v.draw_instance_predictions(output.to("cpu"))
    result_image = result_image.get_image()[:, :, ::-1]
    cv2.imwrite('nms.png', result_image)