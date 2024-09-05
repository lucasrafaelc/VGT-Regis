import pickle
from lxml import etree
from munkres import Munkres


def unidimensional_giou(fig_box, txt_box):
    # case 1: overlap
    if max(fig_box[0], txt_box[0]) < min(fig_box[1], txt_box[1]):
        intersection = min(fig_box[1], txt_box[1]) - max(fig_box[0], txt_box[0])

    # case 2: no overlap
    else:
        intersection = 0

    min_fitting_box = max(fig_box[1], txt_box[1]) - min(fig_box[0], txt_box[0])
    union = (fig_box[1] - fig_box[0]) + (txt_box[1] - txt_box[0]) - intersection

    return (intersection/union) - (min_fitting_box - union)/min_fitting_box


def hungarian_matching(figs, texts):
    optimizer = Munkres()
    cost_matrix = [[1 - unidimensional_giou(fig, text) for text in texts] for fig in figs]
    # print(cost_matrix)
    return optimizer.compute(cost_matrix)


if __name__ == '__main__':
    path = 'result/AAPG-all/AAPG Memoir 77/output.xml'
    page = 17
    root = etree.parse(path).getroot()
    figs = root.xpath(f"//page[@number='{page}']/item[@type='image']")
    texts = root.xpath(f"//page[@number='{page}']/item[@type='text']")

    fig_boxes = [(float(fig.get('y0')), float(fig.get('y1'))) for fig in figs]
    texts_boxes = [(float(txt.get('y0')), float(txt.get('y1'))) for txt in texts]
    solution = hungarian_matching(fig_boxes, texts_boxes)

    for fig_idx, text_idx in solution:
        # print(f"{figs[fig_idx].get('block')} {texts[text_idx].get('block')}")
        print(fig_idx, text_idx)

