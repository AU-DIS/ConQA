
import numpy as np
import ranx
import pandas as pd
from scipy.stats import mannwhitneyu
import os
from tqdm.auto import tqdm
from math import sqrt


def text_to_separable(x):
    return x.replace('gptj6-', 'gptj6*').\
                replace('ViT-L_14_336px', 'ViT*L_14_336px').\
                replace('ViT-B_32', 'ViT*B_32').\
                replace('all-mpnet-base-v2', 'all*mpnet*base*v2').\
                replace('pretrain-large', 'pretrain*large').\
                replace('coco-large', 'coco*large')


def text_from_separable(x):
    return x.replace('*', '-')


def get_datasets():
    ds = set()
    for f in os.listdir('results'):
        if f.startswith('gt-'):
            f = text_to_separable(f)
            n = tuple(f.split('.')[0].split('-')[1:])
            ds.add(n)
    return ds

def get_experiments():
    run = {}
    for f in os.listdir('results'):
        if f.startswith('run-'):
            f = text_to_separable(f)
            n = tuple(f.split('.')[0].split('-')[1:3])
            if n not in run:
                run[n] = set()
            e = f.split('.')[0].split('-')[3]
            run[n].add(e)
    return run


def get_metrics(ds, exps, c_metrics):
    if os.path.exists('results/metrics.npz'):
        metrics = dict(np.load('results/metrics.npz'))
    else:
        metrics = {}
    save = False
    for d in tqdm(ds, leave=False):
        gt = None
        for e in tqdm(exps[d], leave=False, desc=str(d)):
            run = None
            for m in tqdm(c_metrics, leave=False, desc=e):
                name = '-'.join(d + (e, m))
                if name not in metrics:
                    save = True
                    if gt is None:
                        gt = ranx.Qrels.from_parquet('results/gt-' + text_from_separable('-'.join(d)) + '.parquet')
                    if run is None:
                        run = ranx.Run.from_parquet('results/run-' + text_from_separable('-'.join(d + (e,))) + '.parquet')
                        run.make_comparable(gt)
                    metrics[name] = ranx.evaluate(gt, run, metrics=[m], return_mean=False)
    if save:
        np.savez_compressed('results/metrics.npz', **metrics)
    return metrics


def results_to_table(mets, ds, exps, metrics):
    order = {'abs': 1, 'nonabs': 2, 'coco': 3, 
             'extcoco': 4 , 'gptj6-abs': 5, 'gptj6-nonabs': 6}
    def val_sort(elem):
        if elem.name == 'ds_type':
            elem = elem.apply(lambda x: order[x])
        return elem
    res = []
    for d in ds:
        for e in exps[d]:
            row = [d[0], text_from_separable(d[1]), text_from_separable(e)]
            for m in metrics:
                name = '-'.join(d + (e, m))
                row.append(np.mean(mets[name]))
            res.append(row)
    res = pd.DataFrame(data=res, columns=['ds_size', 'ds_type', 'exp'] + metrics)
    res = res.sort_values(['ds_size', 'exp', 'ds_type'], key=val_sort)
    return res


def hypothesis_test(mets, ds, exps, metrics, hypothesis):
    order = {m: i for i, m in enumerate(metrics)}
    def val_sort(elem):
        if elem.name == 'metrics':
            elem = elem.apply(lambda x: order[x])
        return elem
    res = []
    ds = {d[0] for d in ds}
    exps_n = set()
    for e in exps.values():
        exps_n.update(e)
    exps = exps_n
    for d in ds:
        for e in exps:
            for m in metrics:
                row = [d, e, m]
                for h0, h1 in hypothesis:
                    name0 = '-'.join((d, h0, e, m))
                    name1 = '-'.join((d, h1, e, m))
                    if name0 in mets and name1 in mets:
                        val0 = mets[name0]
                        val1 = mets[name1]
                        pval = mannwhitneyu(val0, val1, alternative='less').pvalue
                        row.append(pval)
                    else:
                        row.append(None)
                res.append(row)
    res = pd.DataFrame(data=res, columns=['ds_size', 'exp', 'metrics'] \
                       + list(map(lambda x: f'{x[0]}<{x[1]}', hypothesis)))
    res = res.sort_values(['ds_size', 'exp', 'metrics'], key=val_sort)
    return res



def cohend(d1, d2):
    # calculate the size of samples
    n1, n2 = len(d1), len(d2)
    # calculate the variance of the samples
    s1, s2 = np.var(d1, ddof=1), np.var(d2, ddof=1)
    # calculate the pooled standard deviation
    s = sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))
    # calculate the means of the samples
    u1, u2 = np.mean(d1), np.mean(d2)
    # calculate the effect size
    return (u1 - u2) / s


def effect_size(mets, ds, exps, metrics, hypothesis):
    order = {m: i for i, m in enumerate(metrics)}
    def val_sort(elem):
        if elem.name == 'metrics':
            elem = elem.apply(lambda x: order[x])
        return elem
    res = []
    ds = {d[0] for d in ds}
    exps_n = set()
    for e in exps.values():
        exps_n.update(e)
    exps = exps_n
    for d in ds:
        for e in exps:
            for m in metrics:
                row = [d, e, m]
                for h0, h1 in hypothesis:
                    name0 = '-'.join((d, h0, e, m))
                    name1 = '-'.join((d, h1, e, m))
                    if name0 in mets and name1 in mets:
                        val1 = mets[name0]
                        val0 = mets[name1]
                        pval = cohend(val0, val1)
                        row.append(pval)
                    else:
                        row.append(None)
                res.append(row)
    res = pd.DataFrame(data=res, columns=['ds_size', 'exp', 'metrics'] \
                       + list(map(lambda x: f'{x[0]}<{x[1]}', hypothesis)))
    res = res.sort_values(['ds_size', 'exp', 'metrics'], key=val_sort)
    return res


def difference(mets, ds, exps, metrics, hypothesis):
    order = {m: i for i, m in enumerate(metrics)}
    def val_sort(elem):
        if elem.name == 'metrics':
            elem = elem.apply(lambda x: order[x])
        return elem
    res = []
    ds = {d[0] for d in ds}
    exps_n = set()
    for e in exps.values():
        exps_n.update(e)
    exps = exps_n
    for d in ds:
        for e in exps:
            for m in metrics:
                row = [d, e, m]
                for h0, h1 in hypothesis:
                    name0 = '-'.join((d, h0, e, m))
                    name1 = '-'.join((d, h1, e, m))
                    if name0 in mets and name1 in mets:
                        val0 = np.mean(mets[name0])
                        val1 = np.mean(mets[name1])
                        val = (val1 - val0) / val0 * 100
                        row.append(val)
                    else:
                        row.append(None)
                res.append(row)
    res = pd.DataFrame(data=res, columns=['ds_size', 'exp', 'metrics'] \
                       + list(map(lambda x: f'{x[0]}<{x[1]}', hypothesis)))
    res = res.sort_values(['ds_size', 'exp', 'metrics'], key=val_sort)
    return res


def difference_abs(mets, ds, exps, metrics, hypothesis):
    order = {m: i for i, m in enumerate(metrics)}
    def val_sort(elem):
        if elem.name == 'metrics':
            elem = elem.apply(lambda x: order[x])
        return elem
    res = []
    ds = {d[0] for d in ds}
    exps_n = set()
    for e in exps.values():
        exps_n.update(e)
    exps = exps_n
    for d in ds:
        for e in exps:
            for m in metrics:
                row = [d, e, m]
                for h0, h1 in hypothesis:
                    name0 = '-'.join((d, h0, e, m))
                    name1 = '-'.join((d, h1, e, m))
                    if name0 in mets and name1 in mets:
                        val0 = np.mean(mets[name0])
                        val1 = np.mean(mets[name1])
                        val = (val1 - val0) 
                        row.append(val)
                    else:
                        row.append(None)
                res.append(row)
    res = pd.DataFrame(data=res, columns=['ds_size', 'exp', 'metrics'] \
                       + list(map(lambda x: f'{x[0]}<{x[1]}', hypothesis)))
    res = res.sort_values(['ds_size', 'exp', 'metrics'], key=val_sort)
    return res


if __name__ == '__main__': 
    metrics = ['ndcg', 'r-precision', 'hit_rate@1', 'hit_rate@5', 'hit_rate@10']

    hypothesis = [('abs', 'nonabs'), ('abs', 'extcoco'), ('nonabs', 'extcoco'),
                ('gptj6*abs', 'gptj6*nonabs'), ('gptj6*abs', 'extcoco'), ('gptj6*nonabs', 'extcoco'),
                ('gptj6*abs', 'abs'), ('gptj6*nonabs', 'nonabs'),
                ('abs', 'gptj6*nonabs')]
    
    ds = get_datasets()
    exps = get_experiments()

    mets = get_metrics(ds, exps, metrics)

    df = results_to_table(mets, ds, exps, metrics)
    df.to_csv('metrics.csv')
    
    df = hypothesis_test(mets, ds, exps, metrics, hypothesis)
    df.to_csv('hypothesis.csv')

    # Not reported
    # df = effect_size(mets, ds, exps, metrics, hypothesis)
    # df.to_csv('cohen_d.csv')
    
    df = difference(mets, ds, exps, metrics, hypothesis)
    df.to_csv('difference.csv')

    # Not reported
    # df = difference_abs(mets, ds, exps, metrics, hypothesis)
    # df.to_csv('difference_abs.csv')