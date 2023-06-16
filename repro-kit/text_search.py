from functools import partial
from typing import Optional, Tuple, List
import torch
import itertools
from sentence_transformers import SentenceTransformer
import os 
import pickle
from tqdm import tqdm


def graph_to_triple_sentences(graph, partial=None,
                            min_nodes=None, max_len=None, 
                            disconected_nodes=False):
    sentences = [] 
    nodes = list(graph.nodes)

    if partial is not None:
        nodes.sort(key=lambda x: graph.out_degree(x), reverse=True)
        part = int(len(nodes) * partial)
        if min_nodes is not None:
            part = max(min_nodes, part)
        nodes = nodes[:part]

    for n in nodes:
        inner_sentences = []
        for s in graph.successors(n):
            inner_sentences.extend(itertools.product(graph.nodes[n]['classes'], \
                                                     graph.edges[(n, s)]['classes'],\
                                                     graph.nodes[s]['classes']))
        if graph.out_degree(n) == 0 and disconected_nodes:
            inner_sentences.extend(map(lambda x: (x,), graph.nodes[n]['classes']))
        inner_sentences = list({' '.join(x) for x in inner_sentences})
        current = ''
        for x in inner_sentences:
            if max_len is not None and len(current) + len(x) + 2 > max_len:
                sentences.append(current.lower())
                current = ''
            else:
                current = current + '. '+ x if len(current) > 0 else x
        if len(current) > 0:
            sentences.append(current.lower())
    return sentences


class TextSearchEngine:

    def __init__(self, model='bert-base-uncased', device=None,
                index_graph_to_sentences=partial(graph_to_triple_sentences, max_len=1024),
                graph_to_sentences=partial(graph_to_triple_sentences, partial=0.5,
                                            min_nodes=1, max_len=1024)):
        if device is not None:
            self.device = device
        else:
            self.device ="cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(model, device=self.device)
        self.trained = False
        self.graph_embeddings = None
        self.graph_idx = None
        self.index_graph_to_sentences = index_graph_to_sentences
        self.graph_to_sentences = graph_to_sentences
        pass
                 
    def index(self, graphs, images_path):
        self.graph_embeddings = []
        self.graph_idx = []
        for idx, graph in tqdm(graphs.items()):
            self.graph_embeddings.append(
                self.model.
                    encode(self.index_graph_to_sentences(graph), 
                            convert_to_tensor=True).
                    mean(axis=0, keepdims=True))
            self.graph_idx.append(idx)
        self.graph_embeddings = torch.concat(self.graph_embeddings, dim=0).to(torch.float32)
        self.graph_embeddings = self.graph_embeddings / self.graph_embeddings.norm(dim=1, keepdim=True)
        self.trained = True
        pass
    
    def save(self, path):
        if not self.trained:
            raise 'The search index is uninitialized'
        if os.path.exists(path) and (not os.path.isdir(path) or len(os.listdir(path)) != 0):
            raise  ValueError('The path should point to an empty directory')
        if not os.path.exists(path):
            os.makedirs(path)
        tensor_path = f'{path}{os.sep}embeddings.pt'
        idx_path = f'{path}{os.sep}index.pic'
        torch.save({'embs': self.graph_embeddings}, tensor_path)
        with open(idx_path, 'wb') as f:
            pickle.dump(self.graph_idx, f)
        pass

    def load(self, path) -> None:
        tensor_path = f'{path}{os.sep}embeddings.pt'
        idx_path = f'{path}{os.sep}index.pic'
        embs = torch.load(tensor_path)['embs'].to(self.device)
        with open(idx_path, 'rb') as f:
            idxs = pickle.load(f)
        self.graph_embeddings = embs
        self.graph_idx = idxs
        self.trained = True
        pass
    
    def search_text(self, sentences: str, k: Optional[int]=None) -> Tuple[List[int], List[float]]:
        sentences = [sentences]
        
        with torch.no_grad(): #Do not need to track the gradients for inferences
            sentences_features = self.model.encode(sentences, convert_to_tensor=True)
            sentences_features = sentences_features / sentences_features.norm(dim=1, keepdim=True)
            cosines = self.graph_embeddings @ sentences_features.T #Images x sentences
            similitud = cosines.mean(dim=1)
            arg_sorted = similitud.argsort(descending=True).cpu().numpy()
            similitud = similitud.cpu().numpy()
        if k is not None:
            arg_sorted = arg_sorted[:k]
        return [self.graph_idx[idx] for idx in arg_sorted], similitud[arg_sorted].tolist()