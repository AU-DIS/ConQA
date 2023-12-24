# Required files

By default, the scripts will attempt to automatically download the required files. However, you can manually download them if required (recommended). 

## MS-COCO

From the MS-COCO, the file `annotations_trainval2017.zip` is required. It contains the captions for the images. It can be found on the [MS-COCO Web Site](https://cocodataset.org/#download)

## MS-COCO-5k

For the replicability expermients, it is also necesary the file `val2014.zip` from the [MS-COCO Web Site](https://cocodataset.org/#download).

In addition, the file with the captions is needed. It can be downloaded from [`caption_datasets.zip`](http://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip).

## ECIR 23
The `ecir23/mscoco/ds_test.cvs` provides the queries for ECIR '23 paper [Scene-Centric vs. Object-Centric Image-Text Cross-Modal Retrieval: A Reproducibility Study](https://link.springer.com/chapter/10.1007/978-3-031-28241-6_5).

This file was extracted for the `CLIP_data.zip` file, which is available in [this repository](https://zenodo.org/record/7298031#.Y2jgU-zMLtV) as stated in the README of the [Scene-Centric vs. Object-Centric Image-Text Cross-Modal Retrieval: A Reproducibility Study](https://github.com/mariyahendriksen/ecir23-object-centric-vs-scene-centric-CMR). Notice that the file can be found in the following path `CLIP_data.zip/data/datasets/mscoco/ds_test.csv`.

## Visual genome

From Visual Genome, the following files are required:

* `images.zip` and `images2.zip`: They contain the Visual Genome images.
* `image_data.json.zip`: It contains metadata about the images. It is needed as it contains a mapping between Visual Genomes ids and MS-COCO ids.
* `objects.json.zip`: It contains the objects found on the images.
* `relationships.json.zip`: It contains relationships among objects in the images. 

They can be found on the [Visual Genome Web site](https://homes.cs.washington.edu/~ranjay/visualgenome/).

Alternatively, the Json files can be found in [Kaggle w/o images](https://www.kaggle.com/datasets/mathurinache/visual-genome). In this case, the files should be zipped separately and placed in the datasets folder. For instance, `objects.json` should be zipped into `objects.json.zip`. The image files are in [`images.zip`](https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip) and [`images2.zip`](https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip), respectively.

An older version of Visual Genome with images can be found on [Kaggle w images](https://www.kaggle.com/datasets/dannywu375/visualgenome). In this case, the folder `VG_100K` must be placed on the `images.zip` file and `VG_100K_2` must be placed on the `images2.zip` file.