import pickle
import numpy as np
import base64
import csv
import sys
import zlib
import time
import mmap
from tqdm import tqdm
import argparse

def prepare_images(path_file):
    conqa_images_coco_id = np.load('conqa_images_coco_id.npy')
        
    in_data = {}
    with open(path_file, "r") as tsv_in_file:
        reader = csv.DictReader(tsv_in_file, delimiter='\t', fieldnames = FIELDNAMES)
        for item in tqdm(reader):
            if int(item['image_id']) in conqa_images_coco_id:
                item['image_id'] = int(item['image_id'])
                item['image_h'] = int(item['image_h'])
                item['image_w'] = int(item['image_w'])   
                item['num_boxes'] = int(item['num_boxes'])
                for field in ['boxes', 'features']:
                    item[field] = np.frombuffer(base64.b64decode(item[field]), 
                        dtype=np.float32).reshape((item['num_boxes'],-1))
                in_data[item['image_id']] = item

    return np.array([in_data[k]['features'] for k in conqa_images_coco_id])
    

if __name__ == '__main__':    
    maxInt = sys.maxsize
    csv.field_size_limit(maxInt//10000000000)

    parser = argparse.ArgumentParser()
    parser.add_argument("--path", help="path to the tsv file")
    
    args = parser.parse_args()
    path = args.path
    
    FIELDNAMES = ['image_id', 'image_w','image_h','num_boxes', 'boxes', 'features']
    in_data = prepare_images(path)
    np.savez('conqa_small_precomp_features_fit', imgs=in_data)