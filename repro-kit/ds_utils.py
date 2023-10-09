import os
import sys
import zipfile
from warnings import filterwarnings, warn
from tqdm import tqdm
import networkx as nx
import json
import requests


def read_images_ids():
    data_path = f'datasets'
    res = {}
    def add_to_dict(zip_path, res):
        with zipfile.ZipFile(zip_path) as zf:
            for fn in zf.namelist():
                if fn.split('/')[-1].split('.')[0] == '':
                    continue
                idx = int(fn.split('/')[-1].split('.')[0]) 
                res[idx]= (zip_path, fn)
        pass

    add_to_dict(f'{data_path}{os.sep}images.zip', res)
    add_to_dict(f'{data_path}{os.sep}images2.zip', res)
    return res


def __add_node(node, skg, expert):
        bb = (node['w'], node['h'], node['x'], node['y'])
        skg.add_node(node['object_id'], bb=bb, classes=node['names'], 
                    confidence=[1.0] * len(node['names']), 
                    experts=expert, synsets=node['synsets'])
        pass


def __add_node_edge(node, skg, expert):
        bb = (node['w'], node['h'], node['x'], node['y'])
        skg.add_node(node['object_id'], bb=bb, classes=node['name'], 
                    confidence=1.0, 
                    experts=expert, synsets=node['synsets'])
        pass


def from_vg_14_data(objects, relationships, remove_duplicated_edges=False, no_warning=True):
    if no_warning:
        filterwarnings('ignore')
    skg = nx.DiGraph()
    if objects is not None:
        assert objects['image_id'] == relationships['image_id'], "Objects' and relationships' image id must match"
    nodes = objects['objects'] if objects is not None else None

    edges = relationships['relationships']
    expert = 'visual_genome_1.4'

    if objects is not None:
        for node in nodes:
            __add_node(node, skg, expert)

    added_edges = set()
    for edge in edges:
        i_edge = (edge['subject']['object_id'], edge['object']['object_id'], edge['predicate'])
        if remove_duplicated_edges and i_edge in added_edges:
            continue
        added_edges.add(i_edge)
        #Workarround missing node definitions in Objects
        if edge['subject']['object_id'] not in skg.nodes:
            o_id = edge['subject']['object_id']
            i_id = objects['image_id']
            warn(f'Missing {o_id} object definition in image {i_id}')
            __add_node_edge(edge['subject'], skg, expert)

        if edge['object']['object_id'] not in skg.nodes:
            o_id = edge['object']['object_id']
            i_id = objects['image_id']
            warn(f'Missing {o_id} object definition in image {i_id}')
            __add_node_edge(edge['object'], skg, expert)
        #End workarround
        skg.add_edge(edge['subject']['object_id'], edge['object']['object_id'], 
                    classes=edge['predicate'], confidence=1.0, 
                    experts=expert, synsets=edge['synsets'])
    if no_warning:
        filterwarnings('default')
    return skg


def read_json_zip(path):
    '''
    Reads a json from a compressed file.
    Parameters
    ----------
    path: path to the zip file.
    '''
    with zipfile.ZipFile(path) as zip_file:
        print('Reading the zip file', file=sys.stderr)
        with zip_file.open(path.split(os.sep)[-1][:-4], mode='r') as data_file:
            data = json.load(data_file)
    return data


def download_data(url, dir_path):
    file_path = dir_path + os.sep + url.split('/')[-1]
    if os.path.exists(file_path):
        return file_path
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    if not os.path.isdir(dir_path):
        raise ValueError(f'{dir_path} is not a directory.')
    with open(file_path, "wb") as sg_zip: 
        response = requests.get(url, stream=True)
        total_length = response.headers.get('content-length')
        print(f'Downloading the file {url}', file=sys.stderr)
        if total_length is None: # no content length header
            sg_zip.write(response.content)
        else:
            total_length = int(total_length)
            with tqdm(total=total_length, bar_format='{n_fmt}/{total_fmt} Bytes [{elapsed}<{remaining}]') as pbar:
                for data in response.iter_content(chunk_size=4096):
                    sg_zip.write(data)
                    pbar.update(len(data))
    return file_path


def load_visual_genome_14(data_path):
    '''
    Returns the Visual Genome 1.2 Graphs in its Json format. 
    Automatically downloads the file if not present.

    Parameters
    ----------
    base_dir: Path to the data directory.

    Returns
    -------
    objects: Objects associated to each image.
    relationship: Relationships associated to each object.
    '''
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    objects_url = 'https://visualgenome.org/static/data/dataset/objects.json.zip' 
    relationships_url = 'https://visualgenome.org/static/data/dataset/relationships.json.zip'
    images_part_1 = 'https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip'
    images_part_2 = 'https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip'
    objects_path = download_data(objects_url, data_path)
    relationships_path = download_data(relationships_url, data_path)
    download_data(images_part_1, data_path)
    download_data(images_part_2, data_path)
    return read_json_zip(objects_path), read_json_zip(relationships_path)


def load_full_vg_14(data_path='datasets', objects=None, relationships=None, remove_duplicated_edges=False):
    kgs = {}
    if relationships is None:
        objects, relationships = load_visual_genome_14(data_path)
    if objects is not None:
        for o, r in tqdm(zip(objects, relationships), total=len(objects)):
            kgs[o['image_id']] = from_vg_14_data(o, r, remove_duplicated_edges)
    else:
        for r in tqdm(relationships):
            kgs[o['image_id']] = from_vg_14_data(None, r, remove_duplicated_edges)
    return kgs


def get_coco_caption():
    '''
    Get the captions for the dataset based on the COCO dataset.
    Parameters
    ----------
    data_path: Path to the data directory.

    Returns
    -------
    captions: image_id -> List of captions
    '''
    # data_path = f'datasets'
    data_path = f'../../../VG1.4'
    coco_url = 'http://images.cocodataset.org/annotations/annotations_trainval2017.zip'
    meta_url = 'https://visualgenome.org/static/data/dataset/image_data.json.zip'
    coco_path = f'{data_path}{os.sep}annotations_trainval2017.zip'
    meta_path = f'{data_path}{os.sep}image_data.json.zip'
    if not os.path.exists(coco_path):
        download_data(coco_url, data_path)
    if not os.path.exists(meta_path):
        download_data(meta_url, data_path)
    captions = {}
    with zipfile.ZipFile(coco_path, 'r') as zf:
        with zf.open('annotations/captions_train2017.json') as f:
            data = json.load(f)
        for obj in data['annotations']:
            if obj['image_id'] not in captions:
                captions[obj['image_id']] = []
            captions[obj['image_id']].append(obj['caption'])
        with zf.open('annotations/captions_val2017.json') as f:
            data = json.load(f)
        for obj in data['annotations']:
            if obj['image_id'] not in captions:
                captions[obj['image_id']] = []
            captions[obj['image_id']].append(obj['caption'])
    meta = read_json_zip(meta_path)
    res = {}
    for m in meta:
        if m['coco_id'] is not None and m['coco_id'] in captions:
            res[m['image_id']] = captions[m['coco_id']]
        else:
            res[m['image_id']] = []
    return res
