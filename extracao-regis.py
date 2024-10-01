import argparse
import cv2
import glob
import json
import logging
import os
import pdf2image
import pickle
import pytesseract
import serverUtils
import subprocess
import time
import torch
import warnings
import zipfile
import utils

from create_grid import return_word_grid, select_tokenizer, create_grid_dict, create_mmocr_grid
from datetime import datetime
from detectron2.config import get_cfg
from labels import labels
from mmocr.apis import MMOCRInferencer
from non_max import non_maximum_suppression
from lxml import etree
from text_extraction import extract_text
from tqdm import tqdm
from VGT.object_detection.ditod import add_vit_config
from VGT.object_detection.ditod.VGTTrainer import DefaultPredictor
warnings.filterwarnings("ignore", category=UserWarning)

## remove o erro de parallelism
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def request_batches(num_batches):
    try:
        # Run the command to request the batch files
        result = subprocess.run(
            ['./request_batches.sh', str(num_batches)],  # Command and arguments
            capture_output=True,  # Capture the output
            text=True,  # Output as text
            check=True  #error catching
        )
        
        # Print the output of the script (batch files downloaded)
        print(f"Script output: {result.stdout}")
        
    except subprocess.CalledProcessError as e:
        # Handle errors in script execution
        logger.exception("ERRO AO REQUISITAR O ARQUIVO DE BATCHES NO SERVIDOR")
        print(f"Error running script: {e}")
        print(f"Script stderr: {e.stderr}")
        

## funções para transformar o pdf em imagem, e imagem para grid
def pdf_to_images(filename, dpi, experiment):
    try:
        pdf_name = os.path.basename(filename).split('.pdf')[0]
        dirname = os.path.join(experiment, pdf_name, 'pages')
        if not os.path.exists(dirname):
            os.makedirs(dirname)

        images = pdf2image.convert_from_path(filename, fmt='png', dpi=dpi)
        for i, image in enumerate(images):
            fp = os.path.join(dirname, f'page_{i}.png')
            image.save(fp)
    except Exception as e:
        logger.exception("ERRO NA FUNÇÃO pdf_to_images")

def image_to_grids(image_path, tokenizer, inferencer):
    try:
        result = inferencer(image_path, return_vis=False)
        tokenizer = select_tokenizer(tokenizer)
        grid = create_mmocr_grid(tokenizer, result)
        if grid is not None:
            save_path = os.path.join(*image_path.split('/')[:-2], 'grids', os.path.basename(image_path).split('.')[0] + '.pkl')
            if not os.path.exists(os.path.dirname(save_path)):
                os.makedirs(os.path.dirname(save_path))
            with open(save_path, 'wb') as file:
                pickle.dump(grid, file)
    except Exception as e:
        logger.exception("ERRO NA FUNÇÃO image_to_grids")

def pdf_to_grids(filename, tokenizer, experiment):
    try:
        pdf_name = os.path.basename(filename).split('.pdf')[0]
        dirname = os.path.join(experiment, pdf_name, 'grids')
        if not os.path.exists(dirname):
            os.makedirs(dirname)

        word_grid = return_word_grid(filename)
        tokenizer = select_tokenizer(tokenizer)
        for i in range(len(word_grid)):
            grid = create_grid_dict(tokenizer, word_grid[i])
            if grid is not None:
                with open(os.path.join(dirname, f'page_{i}.pkl'), 'wb') as file:
                    pickle.dump(grid, file)
    except Exception as e:
        logger.exception("ERRO NA FUNÇÃO pdf_to_grids")
                    
def valid_xml_char_ordinal(c):
    codepoint = ord(c)
    # conditions ordered by presumed frequency
    return (
        0x20 <= codepoint <= 0xD7FF or
        codepoint in (0x9, 0xA, 0xD) or
        0xE000 <= codepoint <= 0xFFFD or
        0x10000 <= codepoint <= 0x10FFFF
        )

def get_credenciais():
    hostname = 'geodigital.inf.ufrgs.br' #Obs: testei no portal da UFRGS
    port = 22  # Default SSH port
    username = "user_vgt"
    password = "#9VGT_User9#"
    
    try:
        # Create SSH client
        ssh_client = serverUtils.create_ssh_client(hostname, port, username, password)
        return ssh_client
    except Exception as e:
        logger.exception("ERRO AO LOGAR NO SERVIDOR")
        
if __name__ == '__main__':
    
    # Cria as pastas da aplicação
    if not os.path.exists("input"):
        os.mkdir("input")
        
    if not os.path.exists("output"):
        os.mkdir("output")
    
    if not os.path.exists("temp"):
        os.mkdir("temp")
    
    
    nome_logger = testes = "temp/" + datetime.now().strftime("%d%m%y%H%M%S") + ".log"
    logger = logging.getLogger(__name__)
    logging.basicConfig(filename=nome_logger, level=logging.INFO, format="{asctime} - {levelname} - {message}", style="{")
        
    # baixa o modelo do MMOCR
    try:
        infer = MMOCRInferencer(det='dbnetpp', rec='svtr-small')
    except Exception as e:
        logger.exception("ERRO AO BAIXAR O MMOCR")

    start_timing = time.time()

    parser = argparse.ArgumentParser(description='script to run VGT on pdf')
    parser.add_argument('--root',
                        type=str,
                        default='input/',
                        help='path to input directory')

    parser.add_argument('--dataset',
                        type=str,
                        default='doclaynet',
                        help='pretrain dataset name: doclaynet or publaynet')

    parser.add_argument('--tokenizer',
                        type=str,
                        default='google-bert/bert-base-uncased',
                        help='tokenizer')

    parser.add_argument('--cfg',
                        help='cfg file path',
                        type=str,
                        default='configs/cascade/doclaynet_VGT_cascade_PTM.yaml')

    parser.add_argument("--opts",
                        help="Modify cfg options using the command-line 'KEY VALUE' pairs",
                        default=[],
                        nargs=argparse.REMAINDER)

    parser.add_argument('--dpi',
                        help='pdf conversion resolution',
                        type=int,
                        default=200)

    parser.add_argument('--output',
                        '-o',
                        help='output folder name',
                        type=str,
                        default='output')

    parser.add_argument('--xml',              
                        help='output xml folder name',
                        type=str,
                        default='output-test')
    
    parser.add_argument('--expand',
                        help='expand bounding box by this value',
                        default=5,
                        type=int)

    parser.add_argument('--preprocess_only',
                        action='store_true',
                        default=False,
                        help='preprocess pdfs to run inference later')

    parser.add_argument('--skip_preprocess',
                        action='store_true',
                        default=False,
                        help='skip preprocess')
    
    parser.add_argument('--threshold',
                        default=0.2,
                        help='threshold que será usado para descartar bounding boxes ruins') 
    
    parser.add_argument('--batches',
                        default=1,
                        help='tamanho dos batches de 1 a 25')

    args = parser.parse_args()
    assert os.path.isdir(args.root), 'The root directory does not exist'
    assert 0 < int(args.batches) <= 25, 'Informe um batch válido entre 1 e 25'
    
    # Deletar o arquivo json do batch e pdfs na máquina local
    utils.delete_folder("input/")
    utils.delete_folder("output/")
    
    # Faz o login no ssh
    ssh_client = get_credenciais()
    
    # Requisita os batches no servidor
    #request_batches(args.batches)  =>Descomentar depois
    
    with open("BatchFilePaths.txt", 'r', encoding="utf-8") as batch_file_paths:
        batches = batch_file_paths.readlines()
    batches = [b.strip() for b in batches]
    
    # Limpa o arquivo de batches
    with open("BatchFilePaths.txt", 'w', encoding="utf-8") as batch_file_paths:
        pass
    print(batches)
        
    for batch in batches:
        logger.info(f"==== INICIANDO A EXTRAÇÃO DA BATCH {batch} ====")
        
        # Obter um arquivo da batch do Geodigital
        try:
            serverUtils.download_file(ssh_client, batch, "temp")
            nome_arquivo = batch.split("/")[-1]
            logger.info(f"Arquivo {nome_arquivo} baixado com sucesso do servidor")
        except Exception as e:
            logger.exception("ERRO AO BAIXAR O ARQUIVO DE BATCH")
            continue
        
        # Abre o arquivo com os metadados da batch
        print(f"temp/{nome_arquivo}")
        try:
            with open(f"temp/{nome_arquivo}", "r", encoding="utf-8") as arq_batch:
                dados_batch = json.load(arq_batch)
        except Exception as e:
            logger.exception("ERRO ABRIR O ARQUIVO DA BATCH")
            continue
            
        for pdf in dados_batch['pdfs']:        
            path = "/" + pdf["path"]
            pages = pdf['pages']
            name = pdf['name']
            collection = pdf['collection']
            encoded_name = pdf["encoded-name"]
            
            logger.info(f"Iniciando a extração do arquivo {encoded_name} -- {name}")
            
            # Baixa o arquivo da variável path e salva em input
            try:
                serverUtils.download_file(ssh_client, path, args.root)
                nome_arquivo = path.split("/")[-1]
                logger.info(f"Arquivo {nome_arquivo} baixado com sucesso do servidor")
            except Exception as e:
                logger.exception("ERRO AO BAIXAR O ARQUIVO DO SERVIDOR")
                continue
            finally:
                pass
              
            # FAZ A EXTRAÇÃO DOS PDFS
            pdfs = glob.glob(os.path.join(args.root, '*.pdf'))
            
            if not args.skip_preprocess:
                # Tenta extrair com o pdfplumber
                extrator = {}
                for pdf_path in tqdm(pdfs):
                    pdf_to_images(pdf_path, args.dpi, args.output)
                    pdf_to_grids(pdf_path, args.tokenizer, args.output)
                    extrator[pdf_path] = "plumber"

                    pdf_name = pdf_path.split('.pdf')[0]
                    pdf_name = pdf_name.split("/")[-1]
                    
                    if len(os.listdir(os.path.join(args.output, pdf_name, "grids"))) == 0:            
                        extrator[pdf_path] = "mmocr"
                
                        # Refaz a extração com o mmocr
                        pdf_name = os.path.basename(pdf_path).split('.pdf')[0]
                        for i, image in enumerate(glob.glob(os.path.join(args.output, pdf_name, 'pages', '*.png'))):
                            image_to_grids(image, args.tokenizer, infer)
                    
            assert not args.preprocess_only, 'skipping inference'

            # Step 1: instantiate config
            cfg = get_cfg()
            add_vit_config(cfg)
            cfg.merge_from_file(args.cfg)

            # Step 2: add model weights URL to config
            cfg.merge_from_list(args.opts)

            # Step 3: set device
            device = "cuda" if torch.cuda.is_available() else "cpu"
            cfg.MODEL.DEVICE = device
            
            # Step 4: define model
            predictor = DefaultPredictor(cfg)

            # Step 6: run inference
            inputs = []
            for pdf_i, pdf in enumerate(pdfs):
                pdf_name = os.path.basename(pdf).split('.pdf')[0]
                images = glob.glob(os.path.join(args.output, pdf_name, 'pages', '*.*'))
                
                # sort by page number
                images = sorted(images, key=lambda x: int(x.split('_')[-1].split('.')[0]))
                
                for i, image_path in enumerate(tqdm(images)):
                    img = cv2.imread(image_path)
                    directory_path = os.path.dirname(os.path.dirname(image_path))
                    grid = os.path.join(directory_path, 'grids', os.path.basename(image_path).split('.')[0]+'.pkl')
                    page = os.path.basename(image_path).split('.')[0]

                    # load or create xml for the current pdf
                    
                    arquivo_xml = encoded_name + ".xml" # CRIA DINAMICAMENTE O NOME DO XML
                    xml_path = os.path.join(directory_path, arquivo_xml)
                    
                    if not os.path.exists(xml_path):
                        # Cria os elementos correspondentes ao xml da REGIS                
                        root = etree.Element('document')
                        
                        meta =etree.SubElement(root, 'metadata')
                
                        docid = etree.SubElement(meta, 'field', name='docid')
                        doc_id = arquivo_xml.rpartition(".")[0]
                        docid.text = doc_id
    
                        filename = etree.SubElement(meta, 'field', name='filename')
                        filename.text = pdf_name + '.pdf'
    
                        filetype = etree.SubElement(meta, 'field', name='filetype')
                        filetype.text = 'PDF'

                        # nativo_digital = etree.SubElement(meta, 'field', name='file_origin')
                        # nativo_digital.text = CODIGO BERNARDO
    
                        num_paginas = etree.SubElement(meta, 'field', name='num_paginas')
                        num_paginas.text = str(pages)
                    else:
                        root = etree.parse(xml_path).getroot()

                    if os.path.exists(grid):
                        # run inference
                        with torch.no_grad():
                            output = predictor(img, grid)["instances"]

                        # save VGT output
                        file_name = os.path.basename(image_path).split('.')[0] + '.pkl'
                        output_path = os.path.join(directory_path, 'outputs', file_name)

                        if not os.path.exists(os.path.dirname(output_path)):
                            os.makedirs(os.path.dirname(output_path))
                        pickle.dump(output, open(output_path, 'wb'))

                        # prepare folders to store cropped bounding boxes from each page
                        cropped_image_dir = os.path.join(directory_path, 'cropped_image')
                        if not os.path.exists(cropped_image_dir):
                            os.makedirs(cropped_image_dir)

                        cropped_table_dir = os.path.join(directory_path, 'cropped_table')
                        if not os.path.exists(cropped_table_dir):
                            os.makedirs(cropped_table_dir)
                        
                        cropped_text_dir = os.path.join(directory_path, 'cropped_text')
                        if not os.path.exists(cropped_text_dir):
                            os.makedirs(cropped_text_dir)

                        page_element = etree.Element("page", number=page.split('_')[-1])

                        # extract bounding boxes
                        try:
                            output, scores = non_maximum_suppression(output, 0.3)
                        except Exception as e:
                            logger.exception("ERRO NA FUNÇÃO non_maximum_suppression")
                            continue
                            
                        assert len(output) == len(scores)
                        for j in range(len(output)):
                            score = scores[j].item()
                            
                            # defines the score limiar to save the text crop
                            if score < args.threshold:
                                continue
                            
                            x1 = int(output[j].pred_boxes.tensor.squeeze()[0].item())
                            y1 = int(output[j].pred_boxes.tensor.squeeze()[1].item())
                            x2 = int(output[j].pred_boxes.tensor.squeeze()[2].item())
                            y2 = int(output[j].pred_boxes.tensor.squeeze()[3].item())
                            
                            
                            image_label = {'doclaynet': 6, 'publaynet': 4}
                            table_label = {'doclaynet': 8, 'publaynet': 3}
                            extrator_texto = ""
                            
                            if output[j].pred_classes.item() == image_label[args.dataset]:
                                figure = img[y1:y2, x1:x2, :]
                                crop_path = os.path.join(cropped_image_dir, f'{page}_box{j}.png')
                                cv2.imwrite(crop_path, figure)
                                box_type = 'image'
                                sub_type = ''
                                element_text = crop_path

                            elif output[j].pred_classes.item() == table_label[args.dataset]:
                                figure = img[y1:y2, x1:x2, :]
                                crop_path = os.path.join(cropped_table_dir, f'{page}_box{j}.png')
                                cv2.imwrite(crop_path, figure)
                                box_type = 'table'
                                sub_type = ''
                                element_text = crop_path
                            
                            else:
                                # text
                                h, w = img.shape[:2]
                                figure = img[max(y1-args.expand, 0): min(y2+args.expand, h),
                                            max(x1-args.expand, 0): min(x2+args.expand, w),
                                            :]

                                crop_path = os.path.join(cropped_text_dir, f'{page}_box{j}.png')
                                cv2.imwrite(crop_path, figure)

                                # [x1, y1, x2, y2]
                                bbox = [x1/w, y1/h, x2/w, y2/h]
                                
                                
                                if extrator[pdf] == "plumber":
                                    # mupdf
                                    element_text = extract_text(pdf, int(page.split('_')[-1]), bbox, args.expand)
                                    extrator_texto = "mupdf"
                                    if element_text == '':
                                        element_text = pytesseract.image_to_string(crop_path, lang='eng+por')
                                        extrator_texto = "tesseract"
                                elif extrator[pdf] == "mmocr":
                                    # tesseract
                                    element_text = pytesseract.image_to_string(crop_path, lang='eng+por')
                                    extrator_texto = "tesseract"
                                    
                                box_type = 'text'
                                sub_type = labels[args.dataset][output[j].pred_classes.item()]

                            # add item to xml
                            item_element = etree.Element("item",
                                                        block=str(j),
                                                        type=box_type,
                                                        subtype=sub_type,
                                                        x0=str(x1),
                                                        y0=str(y1),
                                                        x1=str(x2),
                                                        y1=str(y2),
                                                        confianca=str(score),
                                                        extrator=extrator[pdf],
                                                        extrator_texto=extrator_texto)
                            
                            if element_text is not None:
                                element_text = "".join(c for c in element_text if valid_xml_char_ordinal(c))
                            
                            #TODO Verificar se é a melhor forma de tratar a extração de texto retornar vazio
                            item_element.text = element_text if element_text is not None else ''
                            page_element.append(item_element)

                        # save xml
                        root.append(page_element)
                        tree = etree.ElementTree(root)
                        etree.indent(tree, space="\t", level=0)

                        ## alerações
                        # Ordena os elementos dentro de cada página antes de salvar o XML
                        for page in root.findall('page'):
                            items = []
                            for item in page.findall('item'):
                                item_dict = {
                                    'element': item,
                                    'x0': int(item.get('x0')),
                                    'y0': int(item.get('y0')),
                                    'x1': int(item.get('x1')),
                                    'y1': int(item.get('y1'))
                                }
                                items.append(item_dict)

                            # Ordena os itens: primeiro por y0 (ascendente), depois por x0 (ascendente)
                            items.sort(key=lambda x: (x['y0'], x['x0']))

                            # Remove os itens não ordenados da página
                            for item in page.findall('item'):
                                page.remove(item)

                            # Adiciona os itens na ordem correta
                            for item in items:
                                page.append(item['element'])

                        tree.write(xml_path, pretty_print=True, xml_declaration=True, encoding="UTF-8")

                # Manda o arquivo xml para o Geodigital
                try:
                    serverUtils.upload_file(ssh_client, xml_path, "/home/user_vgt/regis-ext")
                    logger.info("Enviei o arquivo xml para o servidor")
                except Exception as e:
                    logger.exception("ERRO AO MANDAR O ARQUIVO XML")
                    continue
                
                # Compacta a pasta de imagens e tabelas
                path_arquivo_zip = f"temp/{encoded_name}.zip"
                with zipfile.ZipFile(path_arquivo_zip, "w") as zip_imagens_tabelas:
                    caminhos_imagens = [os.path.join(cropped_image_dir, nome) for nome in os.listdir(cropped_image_dir)]
                    for imagem in caminhos_imagens:
                        nome_imagem = imagem.split('/')[-1].split("\\")[-1]
                        zip_imagens_tabelas.write(str(imagem), f"imgs/{nome_imagem}")
                        
                    caminhos_tabelas = [os.path.join(cropped_table_dir, nome) for nome in os.listdir(cropped_table_dir)]
                    for tabela in caminhos_tabelas:
                        nome_tabela = tabela.split('/')[-1].split("\\")[-1]
                        zip_imagens_tabelas.write(str(tabela), f"tables/{nome_tabela}")
                
                # Manda o arquivo de imagens para o Geodigital
                try:
                    serverUtils.upload_file(ssh_client, path_arquivo_zip, "/home/user_vgt/regis-img")
                    logger.info("Enviei o arquivo de imagens para o servidor")
                except Exception as e:
                    logger.exception("ERRO AO MANDAR O ARQUIVO COM AS IMAGENS")
                
                # Mandar o arquivo de logs
                try:
                    serverUtils.upload_file(ssh_client, nome_logger, "/home/user_vgt/regis-logs")
                    logger.info("Enviei o arquivo de log para o servidor")
                except Exception as e:
                    logger.exception("ERRO AO MANDAR O ARQUIVO DE LOGS")
                                        
                # Deletar o arquivo json do batch e pdfs na máquina local
                # Deletar a pasta de arquivos
                utils.delete_folder("input/")
                utils.delete_folder("output/")
                           
    utils.delete_folder("temp/")
    print("Deletei a pasta temp")
    end_timing = time.time()
    logger.info(f"Tempo de execução: {end_timing - start_timing}")
