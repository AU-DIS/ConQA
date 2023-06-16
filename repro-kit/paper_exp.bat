echo off

python experiment.py -z small -s clip -m ViT-L/14@336px -e abs --header
python experiment.py -z small -s clip -m ViT-L/14@336px -e nonabs
python experiment.py -z small -s clip -m ViT-L/14@336px -e gptj6-abs
python experiment.py -z small -s clip -m ViT-L/14@336px -e gptj6-nonabs
python experiment.py -z small -s clip -m ViT-L/14@336px -e coco

python experiment.py -z small -s blip -m ViT-B/32 -e abs 
python experiment.py -z small -s blip -m ViT-B/32 -e nonabs
python experiment.py -z small -s blip -m ViT-B/32 -e gptj6-abs
python experiment.py -z small -s blip -m ViT-B/32 -e gptj6-nonabs
python experiment.py -z small -s blip -m ViT-B/32 -e coco

python experiment.py -z small -s blip2 -m ViT-B/32 -e abs 
python experiment.py -z small -s blip2 -m ViT-B/32 -e nonabs
python experiment.py -z small -s blip2 -m ViT-B/32 -e gptj6-abs
python experiment.py -z small -s blip2 -m ViT-B/32 -e gptj6-nonabs
python experiment.py -z small -s blip2 -m ViT-B/32 -e coco

python experiment.py -z small -s text_graph -m all-mpnet-base-v2 -e abs 
python experiment.py -z small -s text_graph -m all-mpnet-base-v2 -e nonabs
python experiment.py -z small -s text_graph -m all-mpnet-base-v2 -e gptj6-abs
python experiment.py -z small -s text_graph -m all-mpnet-base-v2 -e gptj6-nonabs
python experiment.py -z small -s text_graph -m all-mpnet-base-v2 -e coco
