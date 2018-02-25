import numpy as np
import sys
import pickle
import keras.backend as K
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint

def read_data(path,training):
    print ('Reading data from ',path)
    with open(path,'r', encoding='utf-8') as f:
    
        tags = []
        articles = []
        tags_list = []
        
        f.readline()
        for line in f:
            if training :
                start = line.find('\"')
                end = line.find('\"',start+1)
                tag = line[start+1:end].split(' ')
                article = line[end+2:]
                
                for t in tag :
                    if t not in tags_list:
                        tags_list.append(t)
               
                tags.append(tag)
            else:
                start = line.find(',')
                article = line[start+1:]
            
            articles.append(article)
            
        if training :
            assert len(tags_list) == 38,(len(tags_list))
            assert len(tags) == len(articles)
    return (tags,articles,tags_list)

def main():
	(_, X_test,_) = read_data(sys.argv[1], False)

	tokenizer = Tokenizer()
	tokenizer.word_index = pickle.load(open('bow_word_index.pickle', 'rb'))

	test_bag = tokenizer.texts_to_matrix(X_test, 'count')

	model = Sequential()

	model.add(Dense(512,activation='relu',input_dim=51867))
	model.add(Dropout(0.5))
    
	model.add(Dense(512,activation='relu'))
	model.add(Dropout(0.6))

	model.add(Dense(512,activation='relu'))
	model.add(Dropout(0.8))

	model.add(Dense(38,activation='sigmoid'))

	model.load_weights('bow.hdf5')
	Y_pred = model.predict(test_bag)
	thresh = 0.4
	tag_list = pickle.load(open('label_mapping.pickle', 'rb'))
	with open(sys.argv[2], 'w') as output:
		print ('\"id\",\"tags\"',file=output)
		Y_pred_thresh = (Y_pred > thresh).astype('int')
		for index,labels in enumerate(Y_pred_thresh):
			labels = [ tag_list[i] for i,value in enumerate(labels) if value==1 ]
			labels_original = ' '.join(labels)
			print ('\"%d\",\"%s\"'%(index,labels_original),file=output)


if __name__ == '__main__':
	main()