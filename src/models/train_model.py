from transformers import T5ForConditionalGeneration, T5Tokenizer
import warnings

from datasets import load_dataset, load_metric
import transformers
import datasets
import random
import numpy as np
import pandas as pd
import torch
from IPython.display import display, HTML

torch.manual_seed(420)
np.random.seed(420)
warnings.filterwarnings('ignore')

tokenizer = T5Tokenizer.from_pretrained("hetpandya/t5-small-tapaco")
model = T5ForConditionalGeneration.from_pretrained("hetpandya/t5-small-tapaco")

train_df = pd.read_csv('../data/internal/train.csv')
test_df = pd.read_csv('../data/internal/test.csv')
val_df = pd.read_csv('../data/internal/validation.csv')

from datasets import Dataset, DatasetDict

prefix = "detoxify:"
source = "reference"
target = "translation"
max_input_length = 128
max_target_length = 128
batch_size = 32

def preprocess_function(examples):
    inputs = [prefix + ex for ex in examples[source]]
    targets = [ex for ex in examples[target]]
    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)

    # Setup the tokenizer for targets
    labels = tokenizer(targets, max_length=max_target_length, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tds = Dataset.from_pandas(train_df)
vds = Dataset.from_pandas(val_df)
ds = DatasetDict()
ds['train'] = tds
ds['validation'] = vds
ds = ds.map(preprocess_function, batched=True)


from transformers import AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer

model_name = 't5-small'

args = Seq2SeqTrainingArguments(
    f"{model_name}-finetuned-{source}-to-{target}",
    disable_tqdm=True,
    evaluation_strategy = "epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=1,
    predict_with_generate=True,
    report_to='tensorboard',
)

from transformers import AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

import numpy as np
metric = load_metric("sacrebleu")

# simple postprocessing for text
def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]

    return preds, labels

# compute metrics function to pass to trainer
def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Some simple post-processing
    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

    result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    result = {"bleu": result["score"]}

    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)
    result = {k: round(v, 4) for k, v in result.items()}
    return result

ds_train_pieces = []
ds_val_pieces = []
n = 500
dn = 5
for i in range(0,n,dn):
    ds_train_pieces.append(ds['train'].select(range(i, i+dn)))
    ds_val_pieces.append(ds['validation'].select(range(i, i+dn)))

trainer = Seq2SeqTrainer(
    model,
    args,
    train_dataset=ds['train'],
    eval_dataset=ds['validation'],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)


def training_small_pieces():
    for i in range(len(ds_train_pieces)):
        trainer = Seq2SeqTrainer(
            model,
            args,
            train_dataset=ds_train_pieces[i],
            eval_dataset=ds_val_pieces[i],
            data_collator=data_collator,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics
        )
        trainer.train()
        print(f'{i} out of {len(ds_train_pieces)} trained')

epochs = 5
for epoch in range(epochs):
    print(f"Epoch: {epoch+1} out of {epochs}")
    training_small_pieces()
