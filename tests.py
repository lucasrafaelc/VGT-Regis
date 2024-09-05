import shutil, os
from lxml import etree
import argparse
import pytesseract
from math import ceil
import cv2

def testing():
        parser = argparse.ArgumentParser(description='script to run VGT visual inference on pdf')
        parser.add_argument('--root',
                        type=str,
                        default='result/one',
                        help='path to input directory')

        args = parser.parse_args()
        print(args.root)
        entries = os.listdir(args.root)
        directory_path = os.path.dirname(args.root)
        print(directory_path)
        directories = [entry for entry in entries if os.path.isdir(os.path.join(args.root, entry))]
        for directory in directories:
                print(directory)


def test_pytesseract():
        # crop_path = '/home/vgt/pdf-blocks/result_rib/historias_sem_data_page5/pages/page_0.png'
        crop_path = os.path.join('/home/vgt/pdf-blocks/teste01/cropped_text/page_0_box0.png')

        element_text = pytesseract.image_to_string(crop_path,)
        print(element_text)
 

def split_files_into_n_parts(root_dir, output_dir, n):
    # Get a list of all PDF files in the root directory
    files = [f for f in os.listdir(root_dir) if f.endswith('.pdf')]
    
    # Calculate the number of files per split
    total_files = len(files)
    files_per_split = ceil(total_files / n)
    
    # Create output directories and copy files
    for i in range(n):
        part_dir = os.path.join(output_dir, f'output_{i+1}')
        os.makedirs(part_dir, exist_ok=True)
        
        # Determine the file range for this part
        start_index = i * files_per_split
        end_index = start_index + files_per_split
        
        for file in files[start_index:end_index]:
            shutil.copy(os.path.join(root_dir, file), os.path.join(part_dir, file))

    print(f"Files successfully split into {n} parts.")

# Example usage
root_directory = '../RiB-docs'
output_directory = 'rib-teste'
n_parts = 2

split_files_into_n_parts(root_directory, output_directory, n_parts)

# testing()
#test_pytesseract()