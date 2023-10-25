import warnings

from datasets import load_dataset, load_metric
import transformers
import datasets
import random
import numpy as np
import pandas as pd
import torch
import os
from IPython.display import display, HTML

torch.manual_seed(420)
np.random.seed(420)
warnings.filterwarnings('ignore')

DIR = os.path.abspath(os.curdir)

# DIR = DIR.replace(' ', '\ ')
DIR = DIR.split('/')
PATH = '/'.join(DIR[:-2])
PATH += '/models/t5small_tuned/'


from transformers import T5ForConditionalGeneration, T5Tokenizer

tokenizer = T5Tokenizer.from_pretrained("hetpandya/t5-small-tapaco")
model = T5ForConditionalGeneration.from_pretrained(PATH)

def get_paraphrases(sentence, prefix="paraphrase: ", n_predictions=5, top_k=120, max_length=256,device="cpu"):
    text = prefix + sentence + " </s>"
    encoding = tokenizer.encode_plus(
        text, pad_to_max_length=True, return_tensors="pt"
    )
    input_ids, attention_masks = encoding["input_ids"].to(device), encoding[
        "attention_mask"
    ].to(device)

    model_output = model.generate(
        input_ids=input_ids,
        attention_mask=attention_masks,
        do_sample=True,
        max_length=max_length,
        top_k=top_k,
        top_p=0.98,
        early_stopping=True,
        num_return_sequences=n_predictions,
    )

    outputs = []
    for output in model_output:
        generated_sent = tokenizer.decode(
            output, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        if (
                generated_sent.lower() != sentence.lower()
                and generated_sent not in outputs
        ):
            outputs.append(generated_sent)
    return outputs

paraphrases = get_paraphrases("I bought a fucking good pen")

for sent in paraphrases:
    print(sent)