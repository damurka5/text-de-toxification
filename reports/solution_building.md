# Baseline: Dictionary based
Replacing rude words with their synonyms is not a good strategy due to lose of context.

# Hypothesis 1: LSTM model from scratch
Such approach loses context and is hard to train. It took a lot of time on my local machine and does not guarantee good results, so I discarded this model after several trainings steps.

# Hypothesis 2: Bert fine-tuning
Bert is a large language model. It includes attention mechanism, so it understands the context, but the local training of such large model requires a lot of time. I'm a beginner at LLM and faced some problems, so I tried to next solution.

# Hypothesis 3: T5 small paraphraser fine-tuning
Pre-trained T5 model from [Hugging Face](https://huggingface.co/mrm8488/t5-small-finetuned-quora-for-paraphrasing) is lighter than Bert, so it was easier to train. It produced good results even being trained on small piece of dataset. 

# Results
My final model is t5small fine-tuned. It provides good results even it has been trained on a very small dataset due to local computational resources. On a figure below you can see bar-chart with excluded words after detoxification process.
![Fig. 2](figures/bar_plot.png "Fig. 2")
