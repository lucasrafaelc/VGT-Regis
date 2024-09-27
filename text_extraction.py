from lxml import etree
import cv2
import fitz


def extract_text_intersects(pdf_path, page_num, bbox):
    document = fitz.open(pdf_path)
    page = document[page_num]
    w = page.rect.width
    h = page.rect.height
    # print(f'fitz height and width: {h} and {w}')
    bbox = [bbox[0]*w, bbox[1]*h, bbox[2]*w, bbox[3]*h]

    words = page.get_text("words")
    words = [w for w in words if fitz.Rect(w[:4]).intersects(bbox)]
    result_string = ''
    for word in words:
        result_string += word[4] + ' '
    return result_string


def extract_text(pdf_path, page_num, bbox, expand=0):
    document = fitz.open(pdf_path)
    page = document[page_num]
    w = page.rect.width
    h = page.rect.height
    bbox = [bbox[0] * w - expand, bbox[1] * h - expand, bbox[2] * w + expand, bbox[3] * h + expand]
    return page.get_textbox(bbox)


if __name__ == '__main__':
    pdf = 'pdfs/AAPG-109/3 Rock Fragments.pdf'
    xml_path = 'result/AAPG-ALL/3 Rock Fragments/output.xml'
    page_num = 6
    block_num = 2
    root = etree.parse(xml_path).getroot()
    block = root.xpath(f"//page[@number='{page_num}']/item[@block='{block_num}']")[0]

    page_png = cv2.imread('result/AAPG-ALL/3 Rock Fragments/pages/page_6.png')
    h, w = page_png.shape[:2]
    print(f'original height and width: {h} {w}')

    bbox = [int(block.get('x0')) / w,
            int(block.get('y0')) / h,
            int(block.get('x1')) / w,
            int(block.get('y1')) / h]

    text1 = extract_text(pdf, page_num, bbox, 5)
    if text1 == '':
        print('No text found')
    print('')
    print(text1)

