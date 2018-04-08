import time
import nltk
from nltk.corpus import stopwords
import pickle
import json
import random

traintest = 0.7
trainDataFile = './train_data.json'
wordsFileName = './store_words.json'

pickleFile = './naive_bayes_pickle'

def writeToFile(fileName, data_in):
    with open(fileName, 'w') as fp:
        json.dump(data_in, fp)

label_category = ['pos','neg']
stops=stopwords.words("english")
stops.extend(['.', ',', "'s", "n't", '!', '(', ')', '-', ':', '!', '?', '...', '..', '+', ';', '<', '>'])
stop_words = set(stops)
a=set(['didnt', 'didn', 'shouldn', 'mightn','weren', 'not','nor','no','aren','haven','isn','doesn', 'hasn', 'wasn', 'mustn',])
stop_words_modified=list(stop_words-a)
start_time = time.time()

dataset = {}

with open(trainDataFile, 'r') as fp:
    data = json.load(fp)
    document_tuple=[]
    all_words = []
    for label in label_category:
        tokenized_sentences = []
        for list_item in data:
            if (list_item['label']==label):
                s = list_item['text']
                s=s.lower()
                tokenized_sentence_stop = nltk.word_tokenize(s)
                tokenized_sentence = [wd for wd in tokenized_sentence_stop if not wd in stop_words]
                all_words.extend(tokenized_sentence)
                document_tuple.append((tokenized_sentence, label))
    random.shuffle(document_tuple)
    all_words=nltk.FreqDist(all_words)
    word_features = list(all_words.keys())[:5000]
    dataset=word_features
    def find_features (document):
        words = set(document)
        features = {}
        for w in word_features:
            features[w]=(w in words)
        return features

    featuresets = [(find_features(rev), label) for (rev, label) in document_tuple]
    traintestdiv=int(traintest*(len(featuresets)))
    print(len(featuresets))
    print(traintestdiv)
    training_set = featuresets[traintestdiv:]
    testing_set = featuresets[:traintestdiv]
    classifier = nltk.NaiveBayesClassifier.train(training_set)

    print("Naive Bayes Algorithm Accuracy ", (nltk.classify.accuracy(classifier, testing_set))*100)
    classifier.show_most_informative_features(10)

    save_classifier = open(pickleFile,"wb")
    pickle.dump(classifier, save_classifier)
    save_classifier.close()

writeToFile(wordsFileName,dataset)
