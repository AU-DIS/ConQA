python experiment.py -z small -s clip -m ViT-L/14@336px -e abs --header --save_experiment
python experiment.py -z small -s clip -m ViT-L/14@336px -e nonabs --save_experiment
python experiment.py -z small -s clip -m ViT-L/14@336px -e gptj6-abs --save_experiment
python experiment.py -z small -s clip -m ViT-L/14@336px -e gptj6-nonabs --save_experiment
python experiment.py -z small -s clip -m ViT-L/14@336px -e extcoco --save_experiment

python experiment.py -z small -s sgraf -e abs --save_experiment
python experiment.py -z small -s sgraf -e nonabs --save_experiment
python experiment.py -z small -s sgraf -e gptj6-abs --save_experiment
python experiment.py -z small -s sgraf -e gptj6-nonabs --save_experiment
python experiment.py -z small -s sgraf -e extcoco --save_experiment

python experiment.py -z small -s naaf -e abs --save_experiment
python experiment.py -z small -s naaf -e nonabs --save_experiment
python experiment.py -z small -s naaf -e gptj6-abs --save_experiment
python experiment.py -z small -s naaf -e gptj6-nonabs --save_experiment
python experiment.py -z small -s naaf -e extcoco --save_experiment

python experiment.py -z small -s bliprr -m pretrain-large -e abs --save_experiment
python experiment.py -z small -s bliprr -m pretrain-large -e nonabs --save_experiment
python experiment.py -z small -s bliprr -m pretrain-large -e gptj6-abs --save_experiment
python experiment.py -z small -s bliprr -m pretrain-large -e gptj6-nonabs --save_experiment
python experiment.py -z small -s bliprr -m pretrain-large -e extcoco --save_experiment

python experiment.py -z small -s blip2rr -m pretrain -e abs --save_experiment
python experiment.py -z small -s blip2rr -m pretrain -e nonabs --save_experiment
python experiment.py -z small -s blip2rr -m pretrain -e gptj6-abs --save_experiment
python experiment.py -z small -s blip2rr -m pretrain -e gptj6-nonabs --save_experiment
python experiment.py -z small -s blip2rr -m pretrain -e extcoco --save_experiment

echo Analyzing results
python Results_analysis.py
