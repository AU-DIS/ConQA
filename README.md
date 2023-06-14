# ConQA description

ConQA is a crowd-sourced dataset extending the Visual Genome ids for the images. The goal of the dataset is to provide a benchmark for the image retrieval task. The dataset consists of 80 queries divided into 50 conceptual and 30 descriptive queries. A descriptive query mentions some of the objects in the image, for instance, `people chopping' vegetables`. 
While, a conceptual query does not mention objects or only refers to objects in a general context, e.g., `working class life`. More information about the dataset can be found in the paper.

Images and scene graphs can be downloaded from the [Visual Genome Download page](http://visualgenome.org/api/v0/api_home.html). For creating the dataset, we used the most up-to-date files up to version 1.4. The mapping between Visual Genome ids and MS-Coco ids can be found in the file [image_data.json.zip](http://visualgenome.org/static/data/dataset/image_data.json.zip).

The captions for the images are available in the [MS-Coco download site](https://cocodataset.org/#download). We used the 2017 Train/Val annotations.

## Files:

* vg_subset.txt: a list of the Visual Genome Ids used for creating this dataset.
* seed.json: a Json file representing a dictionary {QueryId: QueryData}.
    * QueryData: a dictionary containing:
        * **Text**: text of the query. 
        * **Conceptual**: a boolean indicating if the query is a conceptual query.
        * **Seed**: list of images Visual Genome ids used for seeding the Mturk.
* mturk.json: a Json file representing a dictionary {QueryId: TaskData}. 
    * TaskData is a dictionary {vgId: Relevance}
        * **vgId**: Visual Genome Id of the image.
        * **Relevance**: is a three-element array representing the number of votes for Relevant, Non Relevant, and Unsure, respectively.

## Citation

```
@misc{
    title={ConQA: Conceptual query answering in text-to-image retrieval},
    authors={Rodriguez, Juan Manuel and Tavassoli, Nima and
            Lissandrini, Matteo and Mottin, Davide and
            Levy, Eliezer and Lederman, Gil and Sivov, Dima},
    year={2023}
}
```
