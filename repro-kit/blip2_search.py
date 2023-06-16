import torch
import torch.nn.functional as F
import zipfile
import os

import pickle

from lavis.models import load_model_and_preprocess
from typing import Tuple, Dict, List, Optional
from tqdm.auto import tqdm
from PIL import Image


def load_default_blip2_model():
    model, vis_processors, x = load_model_and_preprocess(name="blip2_image_text_matching", model_type="pretrain", is_eval=True, device="cpu")
    return model, vis_processors["eval"]
    
    
class FastBLIP2ITCSearchEngine:
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

    def __init__(self, model, transform, indexing_device=None, inference_device=None):
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
        self.transform = transform
        self.trained = False
        self.search_batch = 100
        pass
        
    def __process_batch(self, batch):
        with torch.no_grad(): #Do not need to track the gradients for inferences
            image = torch.vstack(batch).to(self.indexing_device)
            with torch.cuda.amp.autocast(enabled=(self.model.device != torch.device("cpu"))):
                image_embeds = self.model.ln_vision(self.model.visual_encoder(image))
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(self.indexing_device)
            query_tokens = self.model.query_tokens.expand(image_embeds.shape[0], -1, -1)

            query_output =self. model.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )
            image_feats = F.normalize(
                self.model.vision_proj(query_output.last_hidden_state), dim=-1
            )
            self.image_embeddings.append(image_feats.cpu())
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
 
    
    def search_text(self, sentences: str, k: Optional[int]=None) -> Tuple[List[int], List[float]]:
        
        with torch.no_grad(): #Do not need to track the gradients for inferences
            text = self.model.tokenizer(sentences,
                    truncation=True,
                    max_length=self.model.max_txt_len,
                    return_tensors="pt",
            ).to(self.inference_device)
            
            text_output = self.model.Qformer.bert(
                text.input_ids,
                attention_mask=text.attention_mask,
                return_dict=True,
            )
            text_feat = F.normalize(
                self.model.text_proj(text_output.last_hidden_state[:, 0, :]), dim=-1
            )
            similitud = torch.max(torch.sum(self.image_embeddings * text_feat[0, :], dim=-1), dim=1)[0]
            arg_sorted = similitud.argsort(descending=True).cpu().numpy()
            similitud = similitud.cpu().numpy()
        if k is not None:
            arg_sorted = arg_sorted[:k]
        return [self.image_idx[idx] for idx in arg_sorted], similitud[arg_sorted].tolist()
    