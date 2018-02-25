import word2vec
import numpy as np
import nltk
import adjustText
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

word2vec.word2vec('all.txt', 'text.bin', window=10, sample=1e-3, size=500, hs=0, negative=10, cbow=1, verbose=True)

model = word2vec.load('text.bin')

N = 1000

voc = []
vec = []
for v in model.vocab:
    voc.append(v)
    vec.append(model[v])

voc = voc[:N]
vec = np.array(vec)[:N]

tsne = TSNE(n_components=2)
reduced = tsne.fit_transform(vec)

tags = set(['JJ', 'NNP', 'NN', 'NNS'])
punct = set(["'", '.', ':', ";", ',', "?", "!", u"’"])

plt.figure()
texts = []
for i, label in enumerate(voc):
    pos = nltk.pos_tag([label])
    if (label[0].isupper() and len(label) > 1 and pos[0][1] in tags and all(c not in label for c in punct)):
        x, y = reduced[i, :]
        texts.append(plt.text(x, y,label, fontdict={'size' : 6}))
        plt.scatter(x, y)

adjustText.adjust_text(texts, arrowprops=dict(arrowstyle='-', color='k', lw=0.5))

plt.savefig('hp.png', dpi=600, bbox_inches='tight')
