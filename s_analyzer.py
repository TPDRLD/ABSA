# Title : Aspect Based Sentiment Analysis
# Authors : Deepti Ramesh, Tanshi Pradhan, Latika Dua
# Dataset : Trip Advisor dataset

import time
import nltk
import pickle
from nltk.tokenize import sent_tokenize, RegexpTokenizer
from nltk.corpus import stopwords, words
from nltk.stem import WordNetLemmatizer
from matplotlib.cbook import unique
from nltk.corpus import wordnet
import os
import json

# Function for polarity detection
def findpolarity(sentence):
    sentence = sentence.lower()
    tokenized_sentence_stop = nltk.word_tokenize(sentence)
    tokenized_sentence = [word for word in tokenized_sentence_stop if not word in stop_words]
    featureset = find_features(tokenized_sentence)
    label = nltk.NaiveBayesClassifier.classify(classifier, featureset)
    return label

# Function for feature detection
def find_features(document):
    words = set(document)
    features = {}
    for c in word_features:
        features[c] = (c in words)
    return features


# Function to find for aspect terms and corresponding adjective
def find_aspect_terms(wrds={}):
    max_similarity_val = 0
    aspect_word_found = False
    adj_found = False
    for (word1, t1) in wrds:
        for wo in predefined_aspect_terms:
            if (wo == 'Overall Aspect'):
                continue

            wnn = predefined_aspects[wo]
            sim_val = 0
            wn2 = lemmatizer.lemmatize(word1)
            try:
                wone = wordnet.synset(wnn + '.n.01')
                wtwo = wordnet.synset(wn2 + '.n.01')
                sim_val = wone.wup_similarity(wtwo)
            except Exception:
                continue

            if (sim_val > sim_val_threshold and sim_val > max_similarity_val):
                asp_word = wo
                max_sim_val = sim_val
                aspect_word_found = True

        if (t1 in adjective):
            adj_found = True

    if (asp_word_found):
        return asp_word
    elif (adj_found):
        return 'Overall Aspect'
    else:
        asp_word = 'VOID'
        return asp_word

#write to a json file
def writeToJson(fileName, data_inp):
        with open(fileName, 'w') as filepointer:
            json.dump(data_inp, filepointer)

def process_content(tokenize):
    try:
        for wor in tokenize:
            wds = tokenizer_reg.tokenize(wor)
            tgd = nltk.pos_tag(wds)
            aspect_term = find_aspect_terms(tgd)
            return aspect_term
    except Exception as ex:
        print(str(ex))
		
sim_val_threshold = 0.6  #threshold value
path = 'jsonfiles/'
senti_threshold = 3  # 1,2 is negative and 3,4,5 is positive
outFileName = './100408.json'
WordSetFileName = './wordSetData.json'

# aspect terms
aspects = {"Service": "service", "Rooms": "room", "Cleanliness": "cleanliness",
                      "Value": "value",
                      "Location": "location", "Sleep Comfort": "sleep",
                      "Business service (e.g., Free WIFI)": "business", "Reception": "reception"}

stops = stopwords.words("english")
stops.extend(['.', ',', "'s", "n't", '!', '(', ')', '-', ':', '!', '?', '...', '..', '+', ';', '<', '>'])
stop_words = set(stops))
stp_wrds = set(['pruce', 'roon', 'sweet', 'unit', 'cheq', 'bathrom', 'chek', 'locaton'])
stop_words_modified = list(stop_words - w2v)
start_time = time.time()

aspect_terms = aspects.keys()
aspect_term_modified = unique(aspects.values())

# print aspect terms
print("Aspect Terms: ")
print(aspect_terms)
print("aspect terms found: ")
print(aspect_term_modified)

start_time = time.time()
listing = os.listdir(path)

aspect_terms_found = []
word_features = []

with open(WordSetFileName, 'r') as g:
    word_features = json.load(f)

print("---------")
print(word_features)
print("++++++++++")

# Naive bayes classifier load
classifier_g = open(ClassifierFileName, "rb")
classifier = pickle.load(classifier_g)
classifier_g.close()

#Function for cleaning the sentence 
def extract_aspect_term(review_line):
    review_line = review_line.lower()
    review_linewords = review_line.split()
    filtered_review_line = [wd for wd in review_linewords if not wd in stop_words_modified]
    clean_review_line = ' '.join(filtered_review_line)
    sent_tokenizer=sent_tokenize(clean_review_line)
    aspect_term = process_content(sent_tokenizer)
    return aspect_term


data_list=[]
k=0
for infile in listing:
    curr_file = path+infile
    k+=1
    if k < 2:
        with open(curr_file) as data_file:
            data = json.load(data_file)
            reviews = data['Actual Reviews']
            for rev in reviews:
                content = rev['Review Text']
                rating = rev['Ratings']
                aspect_terms_found = [ww for ww in rating]
                sentences = tokenizer.tokenize(content)
                for a in sentences:
                    asp_word_found = extract_aspect_term(s)
                    data={}
                    if(asp_word_found!='VOID' and asp_word_found!='Overall rating' and asp_word_found in aspect_terms_found):
                        label=rating[asp_word_found]
                        data['content']=a
                        data['id'] = 'pos' if int(label) > senti_threshold else 'neg'
                        data_list.append(data)
                    elif (asp_word_found == 'Overall' and 'Overall' in aspect_terms_found):
                        label = rating[asp_word_found]
                        data['content'] = a
                        data['id'] = 'pos' if float(label) > senti_threshold else 'neg'
                        data_list.append(data)


writeToJson(writeFileName,data_list)
