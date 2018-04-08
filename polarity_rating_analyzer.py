import time
import nltk
import os
import json
import pickle
from nltk.tokenize import word_tokenize, sent_tokenize, PunktSentenceTokenizer, RegexpTokenizer
from nltk.corpus import stopwords, words
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import state_union
from matplotlib.cbook import unique
from nltk import pos_tag
from nltk.corpus import wordnet

lemmatizer=WordNetLemmatizer()
tokenizer_reg = RegexpTokenizer(r'\w+')
tokenizer = nltk.data.load("tokenizers/punkt/english.pickle")

similarity_threshold = 0.6
path = 'json/'
sentiment_threshold = 3 #(1 and 2 is negative, 3,4 and 5 positive)
outputFileName = './output_file.json'
wordsFileName = './store_words.json'
pickleFile = './naive_bayes_pickle'

defined_aspects = {"Rooms" : "room", "Cleanliness": "cleanliness",
                      "Value":"value", "Service":"service",
                      "Location": "location", "Sleep Quality":"sleep",
                      "Business service (e.g., internet access)":"internet","Check in / front desk":"reception"}

noun = ["NN", "NNS", "NNP", "NNPS"]
adjective = ["JJ", "JJR", "JJS"]

stops=stopwords.words("english")
stops.extend(['.', ',', "'s", "n't", '!', '(', ')', '-', ':', '!', '?', '...', '..', '+', ';', '<', '>'])
stop_words = set(stops)
a=set(['wasn', 'mustn', 'didnt', 'didn', 'shouldn', 'mightn', 'weren', 'not','nor','no','aren','haven','isn','doesn', 'hasn'])
stop_words_modified=list(stop_words-a)
start_time = time.time()

defined_aspect_terms = defined_aspects.keys()
def_aspect_term_modified=unique(defined_aspects.values())

start_time = time.time()
listing = os.listdir(path)

aspect_terms_found = []
word_features = []

with open(wordsFileName, 'r') as f:
    word_features = json.load(f)

classifier_f = open(pickleFile,"rb")
classifier = pickle.load(classifier_f)
classifier_f.close()


def find_features(document):
    words = set(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)
    return features


def findpolarity(sent):
    sent = sent.lower()
    tokenized_sentence_stop = nltk.word_tokenize(sent)
    tokenized_sentence = [wd for wd in tokenized_sentence_stop if not wd in stop_words]
    featureset=find_features(tokenized_sentence)
    label = nltk.NaiveBayesClassifier.classify(classifier, featureset)
    return  label


def writeToFile(fileName, data_in):
    with open(fileName, 'w') as fp:
        json.dump(data_in, fp)


def aspect_term_extractor(review_line):
    review_line = review_line.lower()
    review_linewords = tokenizer_reg.tokenize(review_line)
    filtered_review_line = [wd for wd in review_linewords if not wd in stop_words_modified]
    print(filtered_review_line)
    max_similarity_value = 0
    asp_word_found = False
    for w1 in filtered_review_line:
        for wn in defined_aspect_terms:
            wnn=defined_aspects[wn]
            similarity_value = 0
            wn2 = lemmatizer.lemmatize(w1)
            try:
                wone = wordnet.synset(wnn + '.n.01')
                wtwo = wordnet.synset(wn2 + '.n.01')
                similarity_value = wone.wup_similarity(wtwo)
            except Exception:
                continue

            if(similarity_value > similarity_threshold and similarity_value > max_similarity_value):
                 asp_word = wn
                 max_similarity_value = similarity_value
                 asp_word_found = True

    if(asp_word_found):
        return asp_word
    else:
        asp_word = 'Null'
        return asp_word


data_list=[]
f=0
for infile in listing:
    current_file = path+infile
    f+=1
    if f < 3 :
        with open(current_file) as data_file:
            data = json.load(data_file)
            reviews = data['Reviews']
            for rev in reviews:
                data = {}
                data_rev = {}
                content = rev['Content']
                data["Content"]=content
                data_rev["Overall"]=findpolarity(content)
                rating = rev['Ratings']
                aspect_terms_found = [ww for ww in rating]
                sentences = tokenizer.tokenize(content)
                for s in sentences:
                    polarity = findpolarity(s)
                    asp_word_found = aspect_term_extractor(s)
                    if(asp_word_found!='Null'):
                        data_rev[asp_word_found]=polarity
                    print(s)
                    print(asp_word_found)
                    print(polarity)
                data["Ratings"]=data_rev
                data_list.append(data)

writeToFile(outputFileName,data_list)
