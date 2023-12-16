python experiment.py -z coco5k -s clip -m ViT-L/14@336px -e coco5k -r hit_rate@1,hit_rate@5,hit_rate@10 --header
python experiment.py -z coco5k -s clip -m ViT-L/14 -e coco5k -r hit_rate@1,hit_rate@5,hit_rate@10 
python experiment.py -z coco5k -s stclip -m ViT-L/14 -e coco5k -r hit_rate@1,hit_rate@5,hit_rate@10 
python experiment.py -z coco5k -s bliprr -m coco-large -e coco5k -r hit_rate@1,hit_rate@5,hit_rate@10
python experiment.py -z coco5k -s blip2rr -m coco -e coco5k -r hit_rate@1,hit_rate@5,hit_rate@10
python experiment.py -z coco5k -s sgraf -e coco5k -r hit_rate@1,hit_rate@5,hit_rate@10
#python experiment.py -z coco5k -s naaf -e coco5k -r hit_rate@1,hit_rate@5,hit_rate@10