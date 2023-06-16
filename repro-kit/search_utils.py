import json
import numpy as np

from ds_utils import read_images_ids

def load_images(idx_file='../vg_subset.txt'):
    images = read_images_ids()
    with open(idx_file, 'rt') as f:
        res = [int(idx.strip()) for idx in f]
        images = {i: images[i] for i in res}
    return images


def get_abstract_queries():
    with open('../seed.json', 'rt', encoding='utf-8') as f:
        data = json.load(f)
    queries = set()
    for q_id, val in data.items():
        if val['Conceptual']:
            queries.add(int(q_id))
    return queries


full_relevant = lambda x: x[0] > 0 and x[1] == 0 and x[2] == 0
sure_relevant = lambda x: x[0] > x[1]
unsure_relevant = lambda x: x[0] >= x[1]

def load_queries_relevants(relevant_def=full_relevant):
    with open('../seed.json', 'rt', encoding='utf-8') as f:
        seeds = json.load(f)
    with open('../mturk.json', 'rt', encoding='utf-8') as f:
        data = json.load(f)
    query = {}
    for q_id, val in seeds.items():
        query[int(q_id)] = val['Text']
    relevants = {}
    for q_id, val in data.items():
        relevants[int(q_id)] = {int(img_idx): l_rels for img_idx, l_rels in val.items()}
    relevant_list = {i: {r for r, v in rels.items() if relevant_def(v)} for i, rels in relevants.items()}
    return query, relevants, relevant_list
                   
                            
def load_manual(json_file='amazonTask/merged_search.json'):
    with open('../seed.json', 'rt', encoding='utf-8') as f:
        data = json.load(f)
    clip_res = {}
    for q_id, val in data.items():
        clip_res[int(q_id)] = set(val['Seeds'])
    return clip_res
                            

def merge_rels(rs, human):
    merge = {idx: rel | human[idx] for idx, rel in rs.items()}
    return merge


def recall(pred, gt, k=None):
    res = []
    for q, ret in pred.items():
        rels = gt[q]
        if len(rels) == 0:
            continue
        if k is None:
            ret = ret[:len(rels)]
        else:
            ret = ret[:k]
        res.append(len({r for r in ret if r in rels}) / len(rels))
    return np.mean(res), np.std(res)


def precision(pred, gt, k=None):
    res = []
    for q, ret in pred.items():
        rels = gt[q]
        if len(rels) == 0:
            continue
        if k is None:
            ret = ret[:len(rels)]
        else:
            ret = ret[:k]
        res.append(len({r for r in ret if r in rels}) / len(ret))
    return np.mean(res), np.std(res)


def dcg(ret, gt):
    return np.sum([1 / np.log2(i) for i, r in enumerate(ret, start=2) if r in gt])


def ndcg(pred, gt, k=None):
    if k is None:
        res = [dcg(ret, gt[q]) / dcg(gt[q], gt[q]) for q, ret in pred.items() if len(gt[q]) > 0]
    else:
        res = [dcg(ret[:k], gt[q]) / dcg(list(gt[q])[:k], gt[q]) for q, ret in pred.items() if len(gt[q]) > 0]
    return np.mean(res), np.std(res)


def filter_tagged_images(images, relevants, rels_q=None):
    val_img = set()
    for tagged in relevants.values():
        val_img.update(tagged.keys())
    if rels_q is not None:
        for r in rels_q.values():
            val_img.update(r)
    return {k: v for k, v in images.items() if k in val_img}


def load_exp_gpt_j6(file='exp_GPT_J6B.json'):
    with open(file, 'r', encoding='utf-8') as f:
        res = json.load(f)
    return res