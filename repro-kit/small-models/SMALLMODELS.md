# Reproducibility kit for NAAF and SGRAF

In this document, you'll find instructions on how to reproduce the reported results of NAAF and SGRAF. Before using these models in the experiments, it's essential to follow these steps.


## Downloading small models
In order to evaluate SGRAF and NAAF, we have to generate similarities between input images and texts using these models, and pass the similarities to the repro kit and run the experiments. To generate similarities, you should begin by downloading the [SGRAF](https://github.com/Paranioar/SGRAF) and [NAAF](https://github.com/crossmodalgroup/naaf) projects. Please ensure that you place these projects in the current (```small-models```) directory. We are using the following commit address of the repositories:
```
NAAF: cafcd6ffd053701c9909ee1f76e3132be955e491
SGRAF: e9ff00b4aa444154e692ba5c54f17c8323a172fb
```

According to our usage of the models, it's necessary to make some modifications to certain files. Therefore, once you have downloaded SGRAF and NAAF, replace the original files in the project with these provided files.
* NAAF
    * ```NAAF-ConQA/test.py```
    * ```NAAF-ConQA/evaluation.py```
    * ```NAAF-ConQA/data.py```
* SGRAF
    * ```SGRAF-ConQA/evaluation.py```
    * ```SGRAF-ConQA/data.py```



### Preparing dataset
All required data are located in the  ```small-models-dataset``` directory.

Regarding the image features, as SGRAF and NAAF follow [SCAN](https://github.com/kuanghuei/SCAN) to obtain image features and vocabularies, we need to download the precomputed image features and filter the specific subset of images required for the experiments. To do so:
1. Download and unzip the zip file (2014 Train/Val Image Features with 36 features per image (fixed)) from [bottom-up-attention](https://github.com/peteanderson80/) repository
2. Run ```filter_scan_dataset.py --path [tsv file]``` providing path to the tsv file. This will generate and save the precomputed features for a subset of images.


#### COCO5K experiment
The following link, from  [SGRAF](https://github.com/Paranioar/SGRAF) repository, provides the image features for the coco5k experiment:
```
https://drive.google.com/drive/u/0/folders/1os1Kr7HeTbh8FajBNegW8rjJf6GIhFqC
```
Download and unzip the folder.

```small-models-dataset``` structure should be as the following:
```
small-models-dataset/
├── data/
|   ├── f30k_precomp/
|   ├── coco_precomp/
├── *
```



## Generating similarities

To generate the similarities for NAAF, download the pretrained weights of the [model](https://drive.google.com/file/d/1e3I5Uk2UGHPql4KLIrQW5L7ek3ih34rh/view?usp=sharing) (Flickr30K) according to the original repo, place it into the project's directory, and run:
```
python test.py --experiment [experiment]
```

To generate the similarities for SGRAF, download the pretrained weights of the models ([Flickr30K_SGRAF](https://drive.google.com/file/d/1OBRIn1-Et49TDu8rk0wgP0wKXlYRk4Uj/view?usp=sharing) and [MSCOCO_SGRAF](https://drive.google.com/file/d/1SpuORBkTte_LqOboTgbYRN5zXhn4M7ag/view?usp=sharing)), unzip the downloaded files, and place the unzipped files into the project's directory. After completing the above steps, run the following:

```
python evaluation.py --experiment [experiment]
```

**NOTE:** It is required to satisfy the requirements specified by SGRAF and NAAF. 


Experiments you can run:
* conqa: all the queries from ConQA.
* gpt: all the rephrased queries for ConQA.
* gpt_abs: conceptual rephrased queries for ConQA.
* gpt_nonabs: descriptive rephrased queries for ConQA.
* coco: the first caption in the MS-COCO dataset as query per image for the images in the ConQA dataset.
* ext_coco: all captions in the MS-COCO dataset as queries per image for the images in the ConQA dataset.
* coco5k: queries and images for the coco5k dataset.

These commands create and store similarity files in ```sims/``` within the current directory. At this point, NAAF and SGRAF are prepared for use in the experimental section.