# Reproducibility kit

This folder contains the reproducibility kit for obtaining the reported results.

## Environment 

Base packages:
```
python 3.9.12 
networkx 3.0
pytorch 1.11.0
torchvision 0.12
sentence_transformer 2.2.2
transformers 4.27.3
numpy 1.23.5
nltk 3.6.7
```

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

**NOTE**: It is required to download Visual Genome and MS-COCO dataset into the folder `datasets` to run the reproducibility kit. More information on the required files can be found in [dataset documentation](datasets/FILES.md)](datasets/FILES.md).

## Running retrieval experiments.

To run the retrieval experiments:

```
python experiment.py -z [dataset] [--add_seeds] -s [technique] -m [model] -e [queries]
```

Parameters:
* dataset:
  * small: index only the tagged images. (Reported results)
  * full: index all the images.
* add_seeds: performs the experiments considering the images tagged as seeds by the researchers. Default: not active.
* technique:
  * clip
  * blip
  * blip2
  * text_graph
* model: model used for the experiment. The values vary depending on the technique. The followings are the models reported by the authors
  * ViT-L/14@336px: clip
  * ViT-B/32: blip and blip2
  * all-mpnet-base-v2: text_graph
* queries:
  * full: all the queries from ConQA.
  * abs: conceptual queries from ConQA.
  * nonabs: descriptive queries from ConQA.
  * coco: the first caption in the MS-COCO dataset as query per image.
  * excoco: all captions in the MS-COCO dataset as queries per image.
  * nonabs: descriptive queries from ConQA.
  * gptj6: all the rephrased queries for ConQA.
  * gptj6-abs: conceptual rephrased queries for ConQA.
  * gptj6-nonabs: descriptive rephrased queries for ConQA.

To run all the reported experiments execute `paper_exp.sh`, all the results are present in `paper_res.csv`.

## Semantic analisis

The notebook `Semantic-Analysis.ipynb` presents the WordNet analysis and the transformer classification perplexity. This notebook requires downloading [WordNet dataset](https://www.nltk.org/data.html) for NLTK.

## Query expansion GPT-J6B

The `expand_GPT_J6B.py` generated the expanded queries using [GPT-J6B](https://huggingface.co/EleutherAI/gpt-j-6b) and generates the `exp_GPT_J6B.json` file.



