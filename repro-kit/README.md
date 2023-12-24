# Reproducibility kit

This folder contains the reproducibility kit for obtaining the reported results.

## Environment 

Base packages:
```
python 3.11.4
networkx 3.1
pytorch 2.0.1
torchvision 0.15.2
sentence_transformer 2.2.2
transformers 4.31.0
numpy 1.24.3
nltk 3.8.1
ranx  0.3.16
```

A full description of all the package used for the experiments and their versions can be found in the Conda environment file `environment.yml`

Installing [CLIP model](https://github.com/openai/CLIP): 
```
pip install ftfy regex tqdm
pip install git+https://github.com/openai/CLIP.git
```

Installing [LAVIS](https://github.com/salesforce/LAVIS) (for BLIP 2)
```
git clone https://github.com/salesforce/LAVIS
cd LAVIS
git checkout 3ac397aa075c3e60b9521b012dda3660e3e35f1e
pip install -e .
cd ..
```

Downloading [BLIP](https://github.com/salesforce/BLIP) (for BLIP)
```
git clone https://github.com/salesforce/BLIP
cd BLIP 
git checkout 3a29b7410476bf5f2ba0955827390eb6ea1f4f9d
cd ..
```

**NOTE**: It is required to download Visual Genome and MS-COCO dataset into the folder `datasets` to run the reproducibility kit. More information on the required files can be found in [dataset documentation](datasets/FILES.md).

## Running retrieval experiments.

To run the retrieval experiments:

```
python experiment.py -z [dataset] [--add_seeds] -s [technique] -m [model] -e [queries]
```

Parameters:
* dataset:
  * small: index only the tagged images.
  * full: index all the images.
  * coco5k: MS-COCO 5k. 
  * ecir23: Dataset for the experiment reported as MS-COCO 5k in ECIR '23 paper [Scene-Centric vs. Object-Centric Image-Text Cross-Modal Retrieval: A Reproducibility Study](https://link.springer.com/chapter/10.1007/978-3-031-28241-6_5). 
* add_seeds: performs the experiments considering the images tagged as seeds by the researchers. Default: not active. coco5k ignores this parameter.
* technique:
  * clip: CLIP model based on the original source code released by OpenIA.
  * stclip: CLIP model based on sentence-transformers released.
  * blip: Blip model using ITC.
  * blip2: Blip2 model using ITC.
  * bliprr: Blip model using ITC re-ranking 128 top results using ITM.
  * blip2rr: Blip2 model using ITC re-ranking 128 top results using ITM.
  * blip2itm: Blip2 model using ITM (very slow).
  * sgraf: See (Small Models)[small-models/SMALLMODELS.md] before running.
  * naaf: See (Small Models)[small-models/SMALLMODELS.md] before running.
  * text_graph
* model: model used by the technique. Different techniques support different models.
  * **clip**: RN50, RN101, RN50x4, RN50x16, RN50x64, ViT-B/32, ViT-B/16, ViT-L/14, ViT-L/14@336px
  * **stclip**: ViT-B/32, ViT-B/16, ViT-L/14
  * **blip and variations**: pretrain, coco, pretrain-large, coco-large
  * **blip2 and variations**: pretrain, coco
  * **sgraf**: ignores this parameter.
  * **naaf**: ignores this parameter.
  * **text_graph**: all-mpnet-base-v2
* queries:
  * full: all the queries from ConQA.
  * abs: conceptual queries from ConQA.
  * nonabs: descriptive queries from ConQA.
  * coco: the first caption in the MS-COCO dataset as query per image for the images in the ConQA dataset.
  * excoco: all captions in the MS-COCO dataset as queries per image for the images in the ConQA dataset.
  * nonabs: descriptive queries from ConQA.
  * gptj6: all the rephrased queries for ConQA.
  * gptj6-abs: conceptual rephrased queries for ConQA.
  * gptj6-nonabs: descriptive rephrased queries for ConQA.
  * coco5k: queries for the coco5k dataset. It can be used only with coco5k dataset.
  * ecir23: queries for the ecir23 dataset. It can be used only with ecir23 dataset.

To run all the reported experiments execute `replicability_exp.sh` and `reproducibility_exp.sh` for the replicability and reproducibilit experiments respectively.

## Semantic analisis

The notebook `Semantic-Analysis.ipynb` presents the WordNet analysis and the transformer classification perplexity. This notebook requires downloading [WordNet dataset](https://www.nltk.org/data.html) for NLTK.

## Query expansion GPT-J6B

The `expand_GPT_J6B.py` generated the expanded queries using [GPT-J6B](https://huggingface.co/EleutherAI/gpt-j-6b) and generates the `exp_GPT_J6B.json` file.



