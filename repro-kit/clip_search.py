import torch
import zipfile
import os
import sys
import pickle
import clip

from typing import Tuple, Dict, List, Optional
from tqdm.auto import tqdm
from PIL import Image
from functools import partial


def load_default_clip_model():
    return clip.load("ViT-B/32")


class CLIPSearchEngine:
    '''
    CLIPSearchEngine returns a list of candidates using a CLIP based similarity.
    Images are indexed using a CLIP Neural Network for getting the embeddings.
    
    Graphs are partialy converted into sentences. It generates senteces for a subset
    of nodes with the largest degrees. It is define as partial and the min_nodes.
    The conversition uses the following algorithm:
    ``
    for node n in subset with largest degree:
        for node s in successors of n:
            sentence = n.class + ' ' + edge[n, s].class + ' ' + s.class
    ``
    Then all the senteces are embedded using CLIP Neural Network.
    The cosine similarity between all the sentences and all images is computed. 
    The final similarity for each image is the average of all cosine distance for 
    that image and all the generates sentences for the graphs.
    '''
    
    def __init__(self, model, preprocess, device=None, 
                prompt=None):
        if device is not None:
            self.device = device
        else:
            self.device ="cuda" if torch.cuda.is_available() else "cpu"
        self.model = model.to(self.device)
        self.model.eval()
        self.preprocess = preprocess
        self.trained = False
        self.image_embeddings = None
        self.image_idx = None
        self.prompt = prompt
        pass
        
    def __process_batch(self, batch):
        with torch.no_grad(): #Do not need to track the gradients for inferences
            batch = torch.vstack(batch).to(self.device)
            image_features = self.model.encode_image(batch)
            self.image_embeddings.append(image_features.cpu())
        pass
                 
    def index(self, images_path: Dict[int, Tuple[str, str]], batch_size=100) -> None:
        self.image_embeddings = []
        self.image_idx = []
        p_bar = tqdm(total=len(images_path))
        
        images_batch = []
        zip_files = {x[0] for x in images_path.values()}
        zip_files = {x: zipfile.ZipFile(x, 'r') for x in zip_files}
        for k, (zip_path, internal_path) in images_path.items():
            if len(images_batch) == batch_size:
                self.__process_batch(images_batch)
                p_bar.update(len(images_batch))
                images_batch.clear()
            
            zf = zip_files[zip_path]
            internal_file = zf.open(internal_path)
            image_data = self.preprocess(Image.open(internal_file))[None,...]
            internal_file.close()
            images_batch.append(image_data)
            self.image_idx.append(k)
        #Last batch  
        for zf in zip_files.values():
            zf.close()
        if len(images_batch) != 0:
            self.__process_batch(images_batch)
            p_bar.update(len(images_batch))
        p_bar.close()
        self.image_embeddings = torch.concat(self.image_embeddings, dim=0).to(torch.float32)
        self.image_embeddings = self.image_embeddings.to(self.device)
        self.image_embeddings = self.image_embeddings / self.image_embeddings.norm(dim=1, keepdim=True)
        self.trained = True
        pass
    
    def load(self, path: str) -> None:
        tensor_path = f'{path}{os.sep}embeddings.pt'
        index_path = f'{path}{os.sep}index.pic'
        embs = torch.load(tensor_path)['embs'].to(self.device)
        with open(index_path, 'rb') as f:
            idxs = pickle.load(f)
        self.image_embeddings = embs
        self.image_idx = idxs
        self.trained = True
        pass
    
    def save(self, path: str) -> None:
        if os.path.exists(path) and (not os.path.isdir(path) or len(os.listdir(path)) != 0):
            raise  ValueError('The path should point to an empty directory')
        if not os.path.exists(path):
            os.makedirs(path)
        tensor_path = f'{path}{os.sep}embeddings.pt'
        index_path = f'{path}{os.sep}index.pic'
        torch.save({'embs': self.image_embeddings}, tensor_path)
        with open(index_path, 'wb') as f:
            pickle.dump(self.image_idx, f)
        pass
    
    def search_text(self, sentences: str, k: Optional[int]=None) -> Tuple[List[int], List[float]]:        
        with torch.no_grad(): #Do not need to track the gradients for inferences
            sentences_features = self.model.encode_text(clip.tokenize(sentences).to(self.device)).to(torch.float32)
            sentences_features = sentences_features / sentences_features.norm(dim=1, keepdim=True)
            cosines = self.image_embeddings @ sentences_features.T #Images x sentences
            similitud = cosines.mean(dim=1)
            arg_sorted = similitud.argsort(descending=True).cpu().numpy()
            similitud = similitud.cpu().numpy()
        if k is not None:
            arg_sorted = arg_sorted[:k]
        return [self.image_idx[idx] for idx in arg_sorted], similitud[arg_sorted].tolist()
    