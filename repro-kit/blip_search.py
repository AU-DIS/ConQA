import torch
import torch.nn.functional as F
import zipfile
import os
import sys

if os.path.realpath('BLIP') not in sys.path:
    sys.path.insert(1, os.path.realpath('BLIP'))

import pickle
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from models.blip_itm import blip_itm

from typing import Tuple, Dict, List, Optional
from tqdm.auto import tqdm
from PIL import Image
from functools import partial


class NotIndexedError:
    pass


def load_default_blip_model():
    image_size = 384
    model_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base.pth'
    model = blip_itm(pretrained=model_url, image_size=image_size, vit='base')
    return model


class BLIPSearchEngine:
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
    
    def __init__(self, model, image_size = 384, indexing_device=None, inference_device=None):
        if indexing_device is not None:
            self.indexing_device = indexing_device
        else:
            self.indexing_device ="cuda" if torch.cuda.is_available() else "cpu"
        if inference_device is not None:
            self.inference_device = inference_device
        else:
            self.inference_device ="cuda" if torch.cuda.is_available() else "cpu"
        self.model = model.to(self.indexing_device)
        self.model.eval()
        self.transform = transforms.Compose([
            transforms.Resize((image_size,image_size),interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        ]) 
        self.trained = False
        self.search_batch = 100
        pass
        
    def __process_batch(self, batch):
        with torch.no_grad(): #Do not need to track the gradients for inferences
            batch = torch.vstack(batch).to(self.indexing_device)
            image_features = self.model.visual_encoder(batch)
            self.image_embeddings.append(image_features.cpu())
        pass
                 
    def index(self, images_path: Dict[int, Tuple[str, str]], batch_size=100) -> None:
        if self.trained:
            self.model = self.model.to(self.indexing_device)
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
            image_data = self.transform(Image.open(internal_file).convert('RGB'))[None,...]
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
        self.image_embeddings = self.image_embeddings.to(self.inference_device)
        self.image_embeddings = self.image_embeddings / self.image_embeddings.norm(dim=1, keepdim=True)
        self.trained = True
        self.model = self.model.to(self.inference_device)
        pass
    
    def load(self, path: str) -> None:
        tensor_path = f'{path}{os.sep}embeddings.pt'
        index_path = f'{path}{os.sep}index.pic'
        embs = torch.load(tensor_path)['embs'].to(self.inference_device)
        with open(index_path, 'rb') as f:
            idxs = pickle.load(f)
        self.image_embeddings = embs
        self.image_idx = idxs
        self.trained = True
        self.model = self.model.to(self.inference_device)
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
    
    def __batch_itm(self, text):
        batch_size = self.search_batch
        full_sim = []
        self.model = self.model.to(self.indexing_device)
        text = text.to(self.indexing_device)
        for i in range(0, self.image_embeddings.size()[0], batch_size):
            img_embs = self.image_embeddings[i:i+batch_size,...].to(self.indexing_device)
            image_atts = torch.ones(img_embs.size()[:-1],dtype=torch.long)
            image_atts = image_atts.to(self.indexing_device)
            similitud = self.model.text_encoder(text.input_ids,
                                       attention_mask = text.attention_mask,
                                       encoder_hidden_states = img_embs,
                                       encoder_attention_mask = image_atts,      
                                       return_dict = True,
                                      )
            similitud = self.model.itm_head(similitud.last_hidden_state[:,0,:])     
            similitud = torch.nn.functional.softmax(similitud, dim=1)[:,1].to(self.inference_device)
            full_sim.append(similitud)
        self.model = self.model.to(self.inference_device)
        return torch.concat(full_sim)
    
    def search_text(self, sentences: str, k: Optional[int]=None, match_head='itm') -> Tuple[List[int], List[float]]:
        if not self.trained:
            raise NotIndexedError('The search index is uninitialized')
        
        with torch.no_grad(): #Do not need to track the gradients for inferences
            text = self.model.tokenizer(sentences, padding='max_length', truncation=True, max_length=35, 
                                  return_tensors="pt").to(self.inference_device) 
            
            
            if match_head=='itm':     
                similitud = self.__batch_itm(text)

            elif match_head=='itc':
                text_output = self.model.text_encoder(text.input_ids, attention_mask=text.attention_mask, 
                                                return_dict=True, mode='text')                     
                image_feat = F.normalize(self.model.vision_proj(self.image_embeddings[:,0,:]),dim=-1)   
                text_feat = F.normalize(self.model.text_proj(text_output.last_hidden_state[:,0,:]),dim=-1)    

                similitud = image_feat @ text_feat.t()
                similitud = similitud[:, 0]
                
            arg_sorted = similitud.argsort(descending=True).cpu().numpy()
            similitud = similitud.cpu().numpy()
        if k is not None:
            arg_sorted = arg_sorted[:k]
        return [self.image_idx[idx] for idx in arg_sorted], similitud[arg_sorted].tolist()
    
    
    
class FastBLIPITCSearchEngine:
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

    def __init__(self, model, image_size = 384, indexing_device=None, inference_device=None):
        if indexing_device is not None:
            self.indexing_device = indexing_device
        else:
            self.indexing_device ="cuda" if torch.cuda.is_available() else "cpu"
        if inference_device is not None:
            self.inference_device = inference_device
        else:
            self.inference_device ="cuda" if torch.cuda.is_available() else "cpu"
        self.model = model.to(self.indexing_device)
        self.model.eval()
        self.transform = transforms.Compose([
            transforms.Resize((image_size,image_size),interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        ]) 
        self.trained = False
        self.search_batch = 100
        pass
        
    def __process_batch(self, batch):
        with torch.no_grad(): #Do not need to track the gradients for inferences
            batch = torch.vstack(batch).to(self.indexing_device)
            image_features = self.model.visual_encoder(batch)
            image_features = F.normalize(self.model.vision_proj(image_features[:,0,:]),dim=-1) 
            self.image_embeddings.append(image_features.cpu())
        pass
                 
    def index(self, images_path: Dict[int, Tuple[str, str]], batch_size=100) -> None:
        if self.trained:
            self.model = self.model.to(self.indexing_device)
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
            image_data = self.transform(Image.open(internal_file).convert('RGB'))[None,...]
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
        self.image_embeddings = self.image_embeddings.to(self.inference_device)
        self.image_embeddings = self.image_embeddings / self.image_embeddings.norm(dim=1, keepdim=True)
        self.trained = True
        self.model = self.model.to(self.inference_device)
        pass
    
    def load(self, path: str) -> None:
        tensor_path = f'{path}{os.sep}embeddings.pt'
        index_path = f'{path}{os.sep}index.pic'
        embs = torch.load(tensor_path)['embs'].to(self.inference_device)
        with open(index_path, 'rb') as f:
            idxs = pickle.load(f)
        self.image_embeddings = embs
        self.image_idx = idxs
        self.trained = True
        self.model = self.model.to(self.inference_device)
        pass
    
    def save(self, path: str) -> None:
        if not self.trained:
            raise NotIndexedError('The search index is uninitialized')
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
        if not self.trained:
            raise NotIndexedError('The search index is uninitialized')
        
        with torch.no_grad(): #Do not need to track the gradients for inferences
            text = self.model.tokenizer(sentences, padding='max_length', truncation=True, max_length=35, 
                                  return_tensors="pt").to(self.inference_device) 
            
            text_output = self.model.text_encoder(text.input_ids, attention_mask=text.attention_mask, 
                                            return_dict=True, mode='text')                     
            image_feat = self.image_embeddings   
            text_feat = F.normalize(self.model.text_proj(text_output.last_hidden_state[:,0,:]),dim=-1)    

            similitud = image_feat @ text_feat.t()
            similitud = similitud[:, 0]
                
            arg_sorted = similitud.argsort(descending=True).cpu().numpy()
            similitud = similitud.cpu().numpy()
        if k is not None:
            arg_sorted = arg_sorted[:k]
        return [self.image_idx[idx] for idx in arg_sorted], similitud[arg_sorted].tolist()
    