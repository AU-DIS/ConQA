import argparse
import os
import sys
from tqdm.auto import tqdm
import clip
fusion_path = os.path.pardir

if os.path.realpath(fusion_path) not in sys.path:
    sys.path.insert(1, os.path.realpath(fusion_path))


from search_utils import load_images, load_queries_relevants,load_manual, merge_rels, filter_tagged_images, get_abstract_queries, load_exp_gpt_j6
from blip_search import FastBLIPITCSearchEngine, load_default_blip_model
from blip2_search import FastBLIP2ITCSearchEngine, FastBLIP2ITMSearchEngine, load_default_blip2_model
from clip_search import CLIPSearchEngine
from text_search import TextSearchEngine
from ds_utils import load_full_vg_14, get_coco_caption

from ranx import Qrels, Run, evaluate

NA = 'n/a'


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

def load_or_train_blip2_itm(images, base):
    search_engine = FastBLIP2ITMSearchEngine(*(load_default_blip2_model()[:2]), inference_device='cuda')
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


def gt_to_qrels(gt):
    #Intern should reduce memory consumption
    gt = {sys.intern(str(q)): {sys.intern(str(r)): 1 for r in rels} for q, rels in gt.items()}
    return Qrels(gt)


def rankings_to_run(rankings):
    #Intern should reduce memory consumption
    rankings = {sys.intern(str(q)):{sys.intern(str(idx)): len(ranking) - i for i, idx in enumerate(ranking)} \
                for q, ranking in rankings.items()}
    return Run(rankings)


def eval(search_engine, queries, gt, ds_size, engine, model, ds_eval, metrics, save_run):
    search = {}
    for idx, text in tqdm(queries.items()):
        search[idx] = search_engine.search_text(text)[0]
    gt = gt_to_qrels(gt)
    search = rankings_to_run(search)
    
    if save_run:
        res_dir = 'results'
        if not os.path.exists(res_dir):
            os.makedirs(res_dir)
        gt_path = f'{res_dir}{os.sep}gt-{ds_size}-{ds_eval}.parquet'
        file_model = model.replace("/", "_").replace("@","_")
        if model != 'n/a':
            run_path = f'{res_dir}{os.sep}run-{ds_size}-{ds_eval}-{engine}+{file_model}.parquet'
            search.name = f'{ds_size}-{ds_eval}-{engine}+{model}'
        else:
            run_path = f'{res_dir}{os.sep}run-{ds_size}-{ds_eval}-{engine}.parquet'
            search.name = f'{ds_size}-{ds_eval}-{engine}'
        if not os.path.exists(gt_path):
            gt.save(gt_path)
        if not os.path.exists(run_path):
            search.save(run_path)

    result = evaluate(gt, search, metrics=metrics)
    print(f'{ds_size}, {engine}, {model}, {ds_eval}, {", ".join([str(result[m]) for m in metrics])}')
    #name = '+'.join(exp_name.split(', ')).replace('/', '-')
    #with open(f'{name}.pk', 'wb') as f:
    #    pickle.dump((search, gt), f)
    #print(f'{ds_size}, {engine}, {model}, {ds_eval}, {nd_1[0]}, {nd_1[1]}, {nd_10[0]}, {nd_10[1]}, {nd_100[0]}, {nd_100[1]}, {nd[0]}, {nd[1]}, {rec_100[0]}, {rec_100[1]}, {rec_200[0]}, {rec_200[1]}, {rec_500[0]}, {rec_500[1]}, {rec_1000[0]}, {rec_1000[1]}')
    
    
    
def main():
    parser = argparse.ArgumentParser(prog = 'experiments', description = 'Runs Experiments')
    parser.add_argument('-z', '--dataset_size', choices=['small', 'full']) 
    parser.add_argument('--add_seeds', action='store_true')
    parser.add_argument('-s', '--search_engine', choices=['clip', 'blip', 'blip2', 'blip2itm', 'text_graph']) 
    parser.add_argument('-m', '--model', 
                        default='ViT-B/32',
                        choices=["RN50", "RN101", "RN50x4", "RN50x16", "RN50x64", "ViT-B/32", "ViT-B/16", "ViT-L/14", "ViT-L/14@336px", "all-mpnet-base-v2"]) 
    parser.add_argument('-e', '--dataset_eval', choices=['full', 'abs', 'nonabs', 'coco', 'extcoco', 'gptj6', 'gptj6-abs', 'gptj6-nonabs'])
    parser.add_argument('-t', '--headers', action='store_true')
    parser.add_argument('-r', '--ranx_metrics', default='ndcg@1,ndcg@10,ndcg@100,ndcg,recall@100,recall@200,recall@500,recall@1000')
    parser.add_argument('--save_experiment', action='store_true')
    args = parser.parse_args()
    ds_size = args.dataset_size
    engine = args.search_engine
    model = args.model
    ds_eval = args.dataset_eval
    headers = args.headers
    metrics = args.ranx_metrics.split(',')
    save_run = args.save_experiment

    print(f'Running: {ds_size}, {engine}, {model}, {ds_eval}', file=sys.stderr)
    imgs, q, _, rs, abstract = load_data(ds_size, args.add_seeds)
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
    elif engine == 'blip2itm':
        base = f'blip2itm_index_{ds_size}'
        if args.add_seeds:
            base = base + '_seeds'
        search_engine = load_or_train_blip2_itm(imgs, base)
    elif engine == 'text_graph':
        base = f'trans_{model.replace("/", "_").replace("@","_")}_index_{ds_size}'
        if args.add_seeds:
            base = base + '_seeds'
        search_engine = load_transformer(imgs, model, base)
    if engine not in {'text_graph', 'clip'}:
        model = NA
    q, rs = get_queries_gt(imgs, q, rs, abstract, ds_eval)
    if headers:
        print('ds size, engine, model, ds_eval, ' + ', '.join(metrics))
    if args.add_seeds:
        ds_eval += '+seeds'
    eval(search_engine, q, rs, ds_size, engine, model, ds_eval, metrics, save_run)
    print('***********************************************************', file=sys.stderr)
    pass

if __name__ == '__main__':
    main()