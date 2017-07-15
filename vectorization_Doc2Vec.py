import numpy as np
import matplotlib.pyplot as plt
from gensim import utils
from gensim.models.doc2vec import LabeledSentence
from gensim.models import Doc2Vec
from random import shuffle
from sklearn import decomposition

def train_Doc2Vec(sentences, min_count, window, size, sample, negative, workers, nb_epochs):
    model=Doc2Vec(min_count=min_count, window=window, size=size, sample=sample, negative=negative, workers=workers)
    model.build_vocab(sentences)
    print('start training')
    for epoch in range(nb_epochs):
        print('starting epoch %d'%(epoch))
        #Important to shuffle between each epoch if we want a good generalization
        shuffle(sentences)
        model.train(sentences, total_examples=46387, epochs=nb_epochs)
    model.save('Doc2Vec models/imdb_%d_%d_%d_%f_%d_%d_%d' %(min_count, window, size, sample, negative, workers, nb_epochs))
    return model

def plotWords(model):
    #get model, we use w2v only

    words_np = []
    #a list of labels (words)
    words_label = []
    for word in model.wv.vocab:
        words_np.append(model[word])
        words_label.append(word)
    print('Added %s words. Shape %s'%(len(words_np),np.shape(words_np)))

    pca = decomposition.PCA(n_components=2)
    pca.fit(words_np)
    reduced= pca.transform(words_np)

    # plt.plot(pca.explained_variance_ratio_)
    for index,vec in enumerate(reduced):
        # print ('%s %s'%(words_label[index],vec))
        if index <100:
            x,y=vec[0],vec[1]
            plt.scatter(x,y)
            plt.annotate(words_label[index],xy=(x,y))
    plt.show()
