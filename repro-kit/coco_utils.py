import zipfile
import json
import os
from ds_utils import download_data


def check_coco(data_path=f'datasets'):
    data_path = f'datasets'
    coco_captions_url = 'http://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip'
    images_url = 'http://images.cocodataset.org/zips/val2014.zip'
    coco_caption_path = f'{data_path}{os.sep}caption_datasets.zip'
    images_path = f'{data_path}{os.sep}val2014.zip'
    if not os.path.exists(coco_caption_path):
        download_data(coco_captions_url, data_path)
    if not os.path.exists(images_path):
        download_data(images_url, data_path)
    pass


def load_coco5k_queries(data_path=f'datasets'):
    check_coco(data_path)
    with zipfile.ZipFile(f'{data_path}/caption_datasets.zip', 'r') as z:
        with z.open('dataset_coco.json', 'r') as coco:
            data = json.load(coco)
            data = [img for img in data['images'] if img['split'] == 'test']
    queries = {}
    rels = {}

    for img in data:
        for s in img['sentences'][:5]:
            pos = len(queries)
            queries[pos] = s['raw']
            rels[pos] = [img['imgid']]
    
    return queries, rels

def load_coco5k_imgs(data_path=f'datasets'):
    check_coco(data_path)
    with zipfile.ZipFile(f'{data_path}/caption_datasets.zip', 'r') as z:
        with z.open('dataset_coco.json', 'r') as coco:
            data = json.load(coco)
            data = [img for img in data['images'] if img['split'] == 'test']

    images = {}
    for img in data:
        images[img['imgid']] = ['datasets/val2014.zip', 'val2014/'+img['filename']]
    return images