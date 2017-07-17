import pandas as pd
from gensim.models.doc2vec import TaggedDocument
from gensim.models import Doc2Vec
import numpy as np
from sklearn import decomposition
import matplotlib.pyplot as plt
from keras.layers import Dense, Activation
from keras.models import Sequential
import keras

variants=pd.read_csv("Data/training_variants")
txt=pd.read_csv("Data/training_text_cleaned")
variants2=pd.read_csv("Data/test_variants")
txt2=pd.read_csv("Data/test_text", sep="||")


def appl(sent):
    if type(sent)==str:
        for i in range(5):
            if (sent[i]=="|"):
                return pd.Series(sent[i+2:])
    return sent

for i in range(5668):
    print i
    txt2.iloc[i,0]=appl(txt2.iloc[i,0])[0]

stopword_set=set(["i","me","my","myself","we","our","ours","ourselves","you","your","yours","yourself","yourselves","he","him","his","himself","she","her","hers","herself","it","its","itself","they","them","their","theirs","themselves","what","which","who","whom","this","that","these","those","am","is","are","was","were","be","been","being","have","has","had","having","do","does","did","doing","a","an","the","and","but","if","or","because","as","until","while","of","at","by","for","with","about","against","between","into","through","during","before","after","above","below","to","from","up","down","in","out","on","off","over","under","again","further","then","once","here","there","when","where","why","how","all","any","both","each","few","more","most","other","some","such","no","nor","not","only","own","same","so","than","too","very","s","t","can","will","just","don","should","now"])

def remove_stopwd(sentence,stop):
    if (type(sentence)!=list):
        result=[]
    else:
        result=[k for k in sentence if k not in stop]
    return result

txt=txt.Text
txt2=txt2.iloc[:,0]

txt=txt.str.replace("[\?\!\"\:\;\.\,\'\(\)\[\]\{\}]", "")
txt=txt.str.lower().str.split()
txt=txt.apply(lambda x: remove_stopwd(x,stopword_set))

txt2=txt2.str.replace("[\?\!\"\:\;\.\,\'\(\)\[\]\{\}]", "")
txt2=txt2.str.lower().str.split()

txt2=txt2.apply(lambda x: remove_stopwd(x,stopword_set))

doc2vec_corpus=[]
for i in range(3321):
    doc2vec_corpus.append(TaggedDocument(txt[i],["Diagnos_Train_"+str(i)]))
for i in range(5668):
    doc2vec_corpus.append(TaggedDocument(txt2[i],["Diagnos_Test_"+str(i)]))

print "Corpus built. Doc2Vec training."

doc2vec_model = Doc2Vec(doc2vec_corpus, dm = 0, alpha=0.1, size= 100, min_alpha=0.025)
doc2vec_model.save("Doc2Vec_on_all_data_alpha_0,1_size_100_minalpha_0,025")


def plotDocs(model):
    docs_np = []
    #a list of labels (words)
    docs_label = []
    for doc in range(len(model)):
        docs_np.append(model[doc])
        docs_label.append(doc)
    print('Added %s words. Shape %s'%(len(docs_np),np.shape(docs_np)))

    pca = decomposition.PCA(n_components=2)
    pca.fit(docs_np)
    reduced= pca.transform(docs_np)

    # plt.plot(pca.explained_variance_ratio_)
    for index,vec in enumerate(reduced):
        # print ('%s %s'%(words_label[index],vec))
        if index <100:
            x,y=vec[0],vec[1]
            plt.scatter(x,y)
            plt.annotate(docs_label[index],xy=(x,y))
    plt.show()


plotDocs(doc2vec_model.docvecs)

x=[]
for i in range(3000):
    x.append(list(doc2vec_model.docvecs[i]))

x_test=[]
for i in range(3000,3300):
    x_test.append(list(doc2vec_model.docvecs[i]))

y_=list(variants.Class)
y=[]


for i in range(len(y_)):
	if y_[i]==1:
		y.append([1,0,0,0,0,0,0,0,0])
	elif y_[i]==2:
		y.append([0,1,0,0,0,0,0,0,0])
	elif y_[i]==3:
		y.append([0,0,1,0,0,0,0,0,0])
	elif y_[i]==4:
		y.append([0,0,0,1,0,0,0,0,0])
	elif y_[i]==5:
		y.append([0,0,0,0,1,0,0,0,0])
	elif y_[i]==6:
		y.append([0,0,0,0,0,1,0,0,0])
	elif y_[i]==7:
		y.append([0,0,0,0,0,0,1,0,0])
	elif y_[i]==8:
		y.append([0,0,0,0,0,0,0,1,0])
	elif y_[i]==9:
		y.append([0,0,0,0,0,0,0,0,1])

y_test=y[3000:3300]


model = Sequential()
model.add(Dense(256, input_dim=100, activation='relu'))
model.add(Dense(126, activation='relu'))
model.add(Dense(9, activation='softmax'))
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

model.fit(np.array(x),y[:3000],epochs=20, batch_size=10,validation_split=.05)


model_json = model.to_json()
with open("model_test.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model_test.h5")
print("Saved model to disk")


results_sized=[]
for i in range(5668):
	results_sized.append(model.predict(np.array([doc2vec_model.docvecs[i+3321]]))[0])

results_proba=[]

for i in range(5668):
	summ=0
	vect=[i]
	for j in range(9):
		summ=summ+results_sized[i][j]
	for j in range(9):
		vect.append(results_sized[i][j]/summ)
	results_proba.append(vect)

pd.DataFrame(results_proba, columns=["ID","class1","class2","class3","class4","class5","class6","class7","class8","class9"]).to_csv("Results/Proba_classes_test.csv", index_col = False)
