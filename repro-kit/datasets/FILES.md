# Required files

By default, the scripts will attempt to automatically download the required files. However, you can manually download them if required (recommended). 

## MS-COCO

From the MS-COCO, the file `annotations_trainval2017.zip` is required. It contains the captions for the images. It can be found on the [MS-COCO Web Site](https://cocodataset.org/#download)

## Visual genome

From Visual Genome, the following files are required:

* `images.zip` and `images2.zip`: They contain the Visual Genome images.
* `image_data.json.zip`: It contains metadata about the images. It is needed as it contains a mapping between Visual Genomes ids and MS-COCO ids.
* `objects.json.zip`: It contains the objects found on the images.
* `relationships.json.zip`: It contains relationships among objects in the images. 

They can be found on the [Visual Genome Web site](http://visualgenome.org).

Alternatively, the Json files can be found in [Kaggle w/o images](https://www.kaggle.com/datasets/mathurinache/visual-genome). In this case, the files should be zipped separately and placed in the datasets folder. For instance, `objects.json` should be zipped into `objects.json.zip`. The image files are in [`images.zip`](https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip) and `images2.zip`, respectively.

An older version of Visual Genome with images can be found on [Kaggle w images](https://www.kaggle.com/datasets/dannywu375/visualgenome). In this case, the folder `VG_100K` must be placed on the `images.zip` file and `VG_100K_2` must be placed on the `images2.zip` file.