import torch
import zipfile
import os
import sys
import pickle
import numpy as np

from typing import Tuple, Dict, List, Optional
from tqdm.auto import tqdm
from PIL import Image
from functools import partial
import json


class SGRAFSearchEngine:
    '''
    SGRAFSearchEngine returns a similarities between all the sentences and all the images calculated by SGRAF method.
    '''
    
    def __init__(self, dataset_eval, device=None, 
                prompt=None):
        if device is not None:
            self.device = device
        else:
            self.device ="cuda" if torch.cuda.is_available() else "cpu"
        self.trained = False
               
        self.dataset_eval = dataset_eval
        self.sims = self.load_sims(self.dataset_eval)
        self.key_to_idx = self.load_key_dict()
        self.image_idx = None
        self.prompt = prompt
        pass
    
        
    def load_key_dict(self):
        with open('key-to-idx.json', 'r') as f:
            return json.load(f)
    
    
    def load_sims(self, dataset_eval):
        if dataset_eval == 'full' or dataset_eval == 'abs' or dataset_eval == 'nonabs':
            return torch.from_numpy(np.load('./small-models/sims/sgraf_conqa_sims.npz')['data'])
        elif dataset_eval == 'coco':
            return torch.from_numpy(np.load('./small-models/sims/sgraf_coco_sims.npz')['data'])
        elif dataset_eval == 'extcoco':
            return torch.from_numpy(np.load('./small-models/sims/sgraf_ext_coco_sims.npz')['data'])
        elif dataset_eval == 'gptj6':
            return torch.from_numpy(np.load('./small-models/sims/sgraf_gpt_sims.npz')['data'])
        elif dataset_eval == 'gptj6-abs':
            return torch.from_numpy(np.load('./small-models/sims/sgraf_gpt_abs_sims.npz')['data'])
        elif dataset_eval == 'gptj6-nonabs':
            return torch.from_numpy(np.load('./small-models/sims/sgraf_gpt_nonabs_sims.npz')['data'])
        elif dataset_eval == 'coco5k':
            return torch.from_numpy(np.load('./small-models/sims/sgraf_coco5k_sims.npz')['data'])
        else:
            raise  ValueError('Invalid dataset for evaluation:' + dataset_eval)
            
            
    
    
    def __process_batch(self, batch):
        with torch.no_grad(): #Do not need to track the gradients for inferences
            batch = torch.vstack(batch).to(self.device)
            image_features = self.model.encode_image(batch)
            self.image_embeddings.append(image_features.cpu())
        pass
                 
    def index(self, images_path: Dict[int, Tuple[str, str]], batch_size=100) -> None:
        # self.image_embeddings = []
        self.image_idx = []
        p_bar = tqdm(total=len(images_path))
        
        images_batch = []
        zip_files = {x[0] for x in images_path.values()}
        zip_files = {x: zipfile.ZipFile(x, 'r') for x in zip_files}
        for k, (zip_path, internal_path) in images_path.items():
            # if len(images_batch) == batch_size:
            #     self.__process_batch(images_batch)
            #     p_bar.update(len(images_batch))
            #     images_batch.clear()
            
            zf = zip_files[zip_path]
            # internal_file = zf.open(internal_path)
            # image_data = self.preprocess(Image.open(internal_file))[None,...]
            # internal_file.close()
            # images_batch.append(image_data)
            self.image_idx.append(k)
        #Last batch  
        for zf in zip_files.values():
            zf.close()
        if len(images_batch) != 0:
            self.__process_batch(images_batch)
            p_bar.update(len(images_batch))
            p_bar.close()
        # self.image_embeddings = torch.concat(self.image_embeddings, dim=0).to(torch.float32)
        # self.image_embeddings = self.image_embeddings.to(self.device)
        # self.image_embeddings = self.image_embeddings / self.image_embeddings.norm(dim=1, keepdim=True)
        # self.trained = True
        print(len(self.image_idx))
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
    
    def search_text(self, sentences: int, k: Optional[int]=None) -> Tuple[List[int], List[float]]:       
        if self.dataset_eval=='full' or self.dataset_eval == 'abs' or self.dataset_eval == 'nonabs': 
            sentences_id = self.key_to_idx[str(k)]
        else:
            sentences_id = k

        with torch.no_grad(): #Do not need to track the gradients for inferences
            cosines = self.sims[:, sentences_id]
            cosines = cosines.reshape(len(cosines), 1)
            similitud = cosines.mean(dim=1)
            arg_sorted = similitud.argsort(descending=True).cpu().numpy()
            similitud = similitud.cpu().numpy()
        return [self.image_idx[idx] for idx in arg_sorted], similitud[arg_sorted].tolist()
    