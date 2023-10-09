import argparse
import os
import sys
from tqdm.auto import tqdm
import clip
fusion_path = os.path.pardir
import numpy as np

if os.path.realpath(fusion_path) not in sys.path:
    sys.path.insert(1, os.path.realpath(fusion_path))


from search_utils import load_images, load_queries_relevants,load_manual, merge_rels, recall, filter_tagged_images, precision, get_abstract_queries, ndcg, load_exp_gpt_j6
# from blip_search import FastBLIPITCSearchEngine, load_default_blip_model
# from blip2_search import FastBLIP2ITCSearchEngine, load_default_blip2_model
from clip_search import CLIPSearchEngine
from text_search import TextSearchEngine
from vsrn_search import VSRNSearchEngine
from vse_inf_search import VSESearchEngine
from sgraf_search import SGRAFSearchEngine
from naaf_search import NAAFSearchEngine
from ds_utils import load_full_vg_14, get_coco_caption
import pickle


def load_data(ds_size, seeds):
    imgs = load_images()
    q, r, rs = load_queries_relevants()
    abstract = get_abstract_queries()
    if not seeds:
        human = load_manual()
        set_human = set()
        for hr in human.values():
            set_human.update(hr)
        imgs = {k: v for k, v in imgs.items() if k not in set_human}
        rs = {k: {sv for sv in v if sv not in set_human} for k, v in rs.items()}
    if ds_size == 'small':
        imgs = filter_tagged_images(imgs, r)
    return imgs, q, r, rs, abstract


def load_or_train_blip(images, base):
    search_engine = FastBLIPITCSearchEngine(load_default_blip_model(), inference_device='cpu')
    if os.path.exists(base):
        search_engine.load(base)
    else:
        search_engine.index(images)
        search_engine.save(base)
    return search_engine


def load_or_train_blip2(images, base):
    search_engine = FastBLIP2ITCSearchEngine(*(load_default_blip2_model()[:2]), inference_device='cuda')
    if os.path.exists(base):
        search_engine.load(base)
    else:
        search_engine.index(images)
        search_engine.save(base)
    return search_engine


def load_or_train_clip(images, model, base):
    search_engine = CLIPSearchEngine(*clip.load(model))
    if os.path.exists(base):
        search_engine.load(base)
    else:
        search_engine.index(images)
        search_engine.save(base)
    return search_engine


def load_transformer(images, model, base):
    search_engine = TextSearchEngine(model=model)
    if os.path.exists(base):
        search_engine.load(base)
    else:
        from ds_utils import load_full_vg_14
        graphs = load_full_vg_14()
        graphs = {k: v for k, v in graphs.items() if k in images}
        search_engine.index(graphs, images)
        search_engine.save(base)
    return search_engine

def load_sgraf(images, ds):
    search_engine = SGRAFSearchEngine(ds)
    search_engine.index(images)
    return search_engine

def load_naaf(images, ds):
    search_engine = NAAFSearchEngine(ds)
    search_engine.index(images)
    return search_engine
    

def get_queries_gt(imgs, q, rs, abstract, ds):
    if ds == 'full':
        return q, rs
    if ds == 'abs':
        nq = {}
        nrs = {}
        for idx, text in q.items():
            if idx not in abstract:
                continue
            nq[idx] = text
            nrs[idx] = rs[idx]
        return nq, nrs
    if ds == 'nonabs':
        nq = {}
        nrs = {}
        for idx, text in q.items():
            if idx in abstract:
                continue
            nq[idx] = text
            nrs[idx] = rs[idx]
        return nq, nrs
    if ds == 'coco':
        q = {}
        rs = {}
        caps = get_coco_caption()
        for i, idx in enumerate(imgs.keys()):
            q[i] = caps[idx][0]
            rs[i] = {idx}
        return q, rs
    if ds == 'extcoco':
        q = {}
        rs = {}
        caps = get_coco_caption()
        i = 0
        for idx in imgs.keys():
            for c in caps[idx]:
                q[i] = c
                rs[i] = {idx}
                i += 1
        return q, rs
    if ds == 'gptj6':
        ext = load_exp_gpt_j6()
        nq = {}
        nrs = {}
        i = 0
        for idx, queries in ext.items():
            idx = int(idx)
            for q in queries[:10]:
                nq[i] = q
                nrs[i] = rs[idx]
                i += 1
        return nq, nrs
    if ds == 'gptj6-abs':
        ext = load_exp_gpt_j6()
        nq = {}
        nrs = {}
        i = 0
        for idx, queries in ext.items():
            idx = int(idx)
            if idx not in abstract:
                continue
            for q in queries[:10]:
                nq[i] = q
                nrs[i] = rs[idx]
                i += 1
        return nq, nrs
    if ds == 'gptj6-nonabs':
        ext = load_exp_gpt_j6()
        nq = {}
        nrs = {}
        i = 0
        for idx, queries in ext.items():
            idx = int(idx)
            if idx in abstract:
                continue
            for q in queries[:10]:
                nq[i] = q
                nrs[i] = rs[idx]
                i += 1
        return nq, nrs


def eval(search_engine, queries, gt, exp_name):
    search = {}

    # if len(queries)== 50:
    #     print('conceptual')
    #     conceptual_id = np.load('../../GraphCLIP-baselines/conceptual_id.npy')
    #     i = 0
    #     for idx, text in tqdm(queries.items()):
    #         search[idx] = search_engine.search_text(conceptual_id[i])[0]
    #         i += 1
    # elif len(queries) == 30:
    #     print('descriptive')
    #     descriptive_id = np.load('../../GraphCLIP-baselines/descriptive_id.npy')
    #     i = 0
    #     for idx, text in tqdm(queries.items()):
    #         search[idx] = search_engine.search_text(descriptive_id[i])[0]
    #         i += 1
    # else:
    i = 0
    
    for idx, text in tqdm(queries.items()):
        search[idx] = search_engine.search_text(text, idx)[0]
        i += 1

    nd = ndcg(search, gt)
    nd_1 = ndcg(search, gt, k=1)
    nd_5 = ndcg(search, gt, k=5)
    nd_10 = ndcg(search, gt, k=10)
    rec_1 = recall(search, gt, k=1)
    rec_5 = recall(search, gt, k=5)
    rec_10 = recall(search, gt, k=10)
    # rec_1000 = recall(search, gt, k=1000)
    #name = '+'.join(exp_name.split(', ')).replace('/', '-')
    #with open(f'{name}.pk', 'wb') as f:
    #    pickle.dump((search, gt), f)
    # print(f'{exp_name}, {nd_1[0]}, {nd_1[1]}, {nd_10[0]}, {nd_10[1]}, {nd_100[0]}, {nd_100[1]}, {nd[0]}, {nd[1]}, {rec_100[0]}, {rec_100[1]}, {rec_200[0]}, {rec_200[1]}, {rec_500[0]}, {rec_500[1]}, {rec_1000[0]}, {rec_1000[1]}')
    
    print(f'{exp_name}, ndcg,  {nd_1[0]}, {nd_5[0]}, {nd_10[0]}')
    print(f'{exp_name}, recall,  {rec_1[0]}, {rec_5[0]}, {rec_10[0]}')
    
    
    
def main():
    parser = argparse.ArgumentParser(prog = 'experiments', description = 'Runs Experiments')
    parser.add_argument('-z', '--dataset_size', choices=['small', 'full']) 
    parser.add_argument('--add_seeds', action='store_true')
    parser.add_argument('-s', '--search_engine', choices=['clip', 'blip', 'blip2', 'text_graph', 'vsrn', 'vse_inf', 'sgraf', 'naaf']) 
    parser.add_argument('-m', '--model', 
                        default='ViT-B/32',
                        choices=["RN50", "RN101", "RN50x4", "RN50x16", "RN50x64", "ViT-B/32", "ViT-B/16", "ViT-L/14", "ViT-L/14@336px", "all-mpnet-base-v2"]) 
    parser.add_argument('-e', '--dataset_eval', choices=['full', 'abs', 'nonabs', 'coco', 'extcoco', 'gptj6', 'gptj6-abs', 'gptj6-nonabs'])
    parser.add_argument('-t', '--headers', action='store_true')
    args = parser.parse_args()
    ds_size = args.dataset_size
    engine = args.search_engine
    model = args.model
    ds_eval = args.dataset_eval
    headers = args.headers

    print(f'Running: {ds_size}, {engine}, {model}, {ds_eval}', file=sys.stderr)
    imgs, q, _, rs, abstract = load_data(ds_size, args.add_seeds)
    
    # with open('conqa-images.pkl', 'wb') as f:
    #     pickle.dump(imgs, f)
    
    with open('conqa-queries.pkl', 'wb') as f:
        pickle.dump(q, f)
    
    print(len(imgs))
    print(len(q))
    print(len(rs))
    if engine == 'clip':
        base = f'clip_{model.replace("/", "_").replace("@","_")}_index_{ds_size}'
        if args.add_seeds:
            base = base + '_seeds'
        search_engine = load_or_train_clip(imgs, model, base)
    elif engine == 'blip':
        base = f'fast_blip_index_{ds_size}'
        if args.add_seeds:
            base = base + '_seeds'
        search_engine = load_or_train_blip(imgs, base)
    elif engine == 'blip2':
        base = f'blip2_index_{ds_size}'
        if args.add_seeds:
            base = base + '_seeds'
        search_engine = load_or_train_blip2(imgs, base)
        
    elif engine == 'sgraf':
        search_engine = load_sgraf(imgs, ds_eval)
        
    elif engine == 'naaf':
        search_engine = load_naaf(imgs, ds_eval)
        
    elif engine == 'text_graph':
        base = f'trans_{model.replace("/", "_").replace("@","_")}_index_{ds_size}'
        if args.add_seeds:
            base = base + '_seeds'
        search_engine = load_transformer(imgs, model, base)

    
    print('###', len(q))
    print('###', np.mean([len(v) for k,v in rs.items()]))
    q, rs = get_queries_gt(imgs, q, rs, abstract, ds_eval)
    print('###', len(q))
    print('###', np.mean([len(v) for k,v in rs.items()]))
    if headers:
        print('exp, ds size, engine, model, ds_eval, NDGC@1, NDGC@1 Std, NDGC@10, NDGC@10 Std, NDGC@100, NDGC@100 Std, NDGC, NDGC Std, R@100, R@100 Std, R@200, R@200 Std, R@500, R@500 Std, R@1000, R@1000 Std')
    eval(search_engine, q, rs, f'exp, {ds_size}, {engine}, {model}, {ds_eval}')
    print('***********************************************************', file=sys.stderr)
    pass

if __name__ == '__main__':
    main()