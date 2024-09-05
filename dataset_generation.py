import argparse
import os
import glob
from tqdm import tqdm
from lxml import etree
from text_image_matching import hungarian_matching
import shutil
import pandas as pd


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir',
                        type=str,
                        default='result/AAPG-ALL',
                        help='directory root where results for each pdf are stored',)

    parser.add_argument('--output_dir',
                        type=str,
                        default='dataset/',
                        help='output directory',)

    args = parser.parse_args()
    output = {
        'image': [],
        'text': [],
        'gt_text': [],
    }

    working_dirs = glob.glob(os.path.join(args.input_dir, '*'))

    os.makedirs(os.path.join(args.output_dir, 'images'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'texts'), exist_ok=True)

    for working_dir in working_dirs:
        root = etree.parse(os.path.join(working_dir, 'output.xml')).getroot()
        # print('Processing {}'.format(working_dir))
        pages = root.xpath(f"//page")

        for j in tqdm(range(len(pages))):
            images = root.xpath(f"//page[@number={j}]/item[@type='image']")
            texts = root.xpath(f"//page[@number='{j}']/item[@type='text']")

            texts_filtered = []
            for txt in texts:
                if txt.text is not None and len(txt.text) > 4:
                    texts_filtered.append(txt)

            fig_boxes = [(float(fig.get('y0')), float(fig.get('y1'))) for fig in images]
            texts_boxes = [(float(txt.get('y0')), float(txt.get('y1'))) for txt in texts_filtered]
            # print(len(fig_boxes), len(texts_boxes))

            if len(texts_boxes) > 0 and len(fig_boxes) > 0:
                for fig_i, text_i in hungarian_matching(fig_boxes, texts_boxes):

                    image_path = os.path.join(working_dir, 'cropped_image',
                                              'page_{}_box{}.png'.format(j, images[fig_i].get('block')))
                    text_path = os.path.join(working_dir, 'cropped_text',
                                             'page_{}_box{}.png'.format(j, texts_filtered[text_i].get('block')))

                    img_dst = os.path.join(args.output_dir, 'images',
                                           '{}_{}_{}.png'.format(os.path.basename(working_dir), j, images[fig_i].get('block')))
                    txt_dst = os.path.join(args.output_dir, 'texts',
                                           '{}_{}_{}.png'.format(os.path.basename(working_dir), j, texts_filtered[text_i].get('block')))

                    shutil.copy2(image_path, img_dst)
                    shutil.copy2(text_path, txt_dst)

                    output['gt_text'].append(texts_filtered[text_i].text)
                    output['image'].append(img_dst)
                    output['text'].append(txt_dst)

    # save output
    data = pd.DataFrame.from_dict(output)
    data.to_excel(os.path.join(args.output_dir, 'dataset.xlsx'))

