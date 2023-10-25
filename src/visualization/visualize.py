import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.append('../')
from src.models.predict_model import *


def uncommon_words(A, B):
    count = {}
    for word in A.split():
        count[word] = count.get(word, 0) + 1

    for word in B.split():
        count[word] = count.get(word, 0) + 1

    return [word for word in count if count[word] == 1]


DIR2 = os.path.abspath(os.curdir)

DIR2 = DIR2.split('/')
PATH2 = '/'.join(DIR2[:-2])
PATH2 += '/data/internal/test.csv'

test_set = pd.read_csv(PATH2)
N = 3000
test_set_slice = test_set[:N]

toxic_sentences = test_set_slice['reference']

paraphrased = []
for sentence in toxic_sentences.tolist():
    translated = get_paraphrases(sentence=sentence, n_predictions=1)
    if len(translated) != 0:
        paraphrased.append(translated[0])
    else:
        paraphrased.append(sentence)
toxic_list = toxic_sentences.tolist()
uncommon_words_counter = {}
meeted_words = []
for i in range(N):
    words = uncommon_words(toxic_list[i], paraphrased[i])
    for wrd in words:
        if len(wrd) > 3:
            if wrd in meeted_words:
                uncommon_words_counter[wrd] = uncommon_words_counter[wrd] + 1
            else:
                uncommon_words_counter[wrd] = 1
                meeted_words.append(wrd)

# print(uncommon_words_counter)
sorted_dict = dict(sorted(uncommon_words_counter.items(), key=lambda item: item[1], reverse=True))
# print(sorted_dict)

keys = list(sorted_dict.keys())
values = list(sorted_dict.values())

DIR3 = os.path.abspath(os.curdir)

DIR3 = DIR3.split('/')
PATH3 = '/'.join(DIR3[:-2])
PATH3 += '/reports/figures/'

fig = plt.figure(figsize =(10, 7))
# creating the bar plot
plt.bar(keys[:10], values[:10])

plt.xlabel("Words")
plt.ylabel("Count")
plt.title("Words replaced by model")
fig.savefig(PATH3+'bar_plot.png')
plt.show()