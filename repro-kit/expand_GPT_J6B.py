from transformers import pipeline, set_seed
from search_utils import load_queries_relevants
from tqdm.auto import tqdm
import json


if __name__ == '__main__':
    generator = pipeline('text-generation', model='EleutherAI/gpt-j-6B')

    q, r, rs = load_queries_relevants()

    res = {}
    for idx, query in tqdm(q.items()):
        set_seed(42)
        l_gen = []
        while len(l_gen) < 10:
            gen = generator(f'"{query}" can be rephrase as "', max_length=30, num_return_sequences=10)
            for g in gen:
                text = g['generated_text'].split('"')
                if len(text) > 4 and text[3] not in l_gen and text[3].strip().lower() != query.strip().lower():
                    l_gen.append(text[3])
                if len(l_gen) == 10:
                    break
        res[idx] = l_gen
    
    with open('exp_GPT_J6B.json', 'w', encoding='utf-8') as f:
        json.dump(res, f)