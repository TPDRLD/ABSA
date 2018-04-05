import time
import nltk
from nltk.corpus import stopwords, words
import pickle
import json
import random
from NaiveBayesClassifier

traintestdevidefactor = 0.6
writeFileName = './trainData.json'
writeWordSetFileName = './wordSetData.json'

label_cat = ['pos','neg']
stops=stopwords.words("english")
stops.extend(['.', ',', "'s", "n't", '!', '(', ')', '-', ':', '!', '?', '...', '..', '+', ';', '<', '>'])
stop_words = set(stops)
a=set(['not','nor','no','aren','haven','isn','doesn', 'hasn', 'wasn', 'mustn', 'didnt', 'didn', 'shouldn', 'mightn','weren'])
stop_words_modified=list(stop_words-a)
start_time = time.time()


# Writing trained data to json files
def writeToJson(fileName, data_in):
    with open(fileName, 'w') as filepointer:
        json.dump(data_in, filepointer)

dataset = {dataset\}

with open(writeFileName, 'r') as filepointer:
    data = json.load(filepointer)
    doc=[]
    aword = []
    for lbl in lbl_cat:
        tokenized_sentences = []
        for list_item in data:
            if (list_item['lbl']==lbl):a
                a = list_item['txt']
                a=a.lower()
                tokenize_stpsent = nltk.word_tokenize(a)
                tokenize_sent = [wod for wod in tokenize_stpsent if not wod in stop_words]
                aword.extend(tokenize_sent)
                doc.append((tokenize_sent, lbl))
    random.shuffle(doc)
    aword=nltk.FreqDist(aword)
    word_features = list(aword.keys())[:7000]
    dataset=wrd_feat
    def find_features (document):
        wrds = set(document)
        features = {}
        for wo in wrd_feat:
            features[wo]=(wo in wrds)
        return features

    featuresets = [(find_features(rev), label) for (rev, label) in doc]
    traintestdevide=int(traintestdevidefactor*(len(featuresets)))
    print(len(featuresets))
    print(traintestdevide)
    training_set = featuresets[traintestdevide:]
    testing_set = featuresets[:traintestdevide]

    # Naive bayes classifier
    classifier = nltk.NaiveBayesClassifier.train(training_set)

    print("Naive Bayes Algorithm Accuracy ", (nltk.classify.accuracy(classifier, testing_set))*100)
    classifier.show_most_informative_features(15)

    writeToJson(writeWordSetFileName,dataset)
