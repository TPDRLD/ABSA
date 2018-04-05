import time
import nltk
from nltk.corpus import state_union
from nltk import pos_tag
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize, sent_tokenize, PunktSentenceTokenizer, RegexpTokenizer
from nltk.corpus import stopwords, words
from nltk.stem import PorterStemmer, WordNetLemmatizer
from word2vec import clean_review, get_similar_words
from matplotlib.cbook import unique
import os
import json

#Function to find for aspect terms and corresponding adjective
def find_aspect_terms(wrds={}):
    maxsimval = 0
    word_found = False
    adjective_found = False
    for (w1, t1) in wrds: 
        for wn in aspect_terms_found:
            if(wn=='Overall'):
                continue

            wnn=aspects[wn]
            sim_val = 0
            wn2 = lemmatize.lemmatize(w1)
            try:
                wone = wordnet.synset(wnn + '.n.01')
                wtwo = wordnet.synset(wn2 + '.n.01')
                sim_val = wone.wup_similarity(wtwo)
            except Exception:
                continue

            if(sim_val > similarity_val_threshold and sim_val > maxsimval):
                 asp_word = wn
                 max_sim_val = sim_val
                 asp_word_found = True

        if(term1 in adjective):
            adj_found=True

    if(word_found): 
        return asp_word
    elif(adjective_found):
        return 'Overall'
    else:
        asp_word = 'VOID'
        return asp_word


    # Function to write to a json file

    def writeToJson(fileName, data_in):
        with open(fileName, 'w') as fp:
            json.dump(data_in, fp)


def extractterm(review_line):
    review_line = review_line.lower()
    review_linewords = review_line.split()
    filtered_review_line = [wd for wd in review_linewords if not wd in stop_words_modified]
    clean_review_line = ' '.join(filtered_review_line)
    sent_tokenizer=sent_tokenize(clean_review_line)
    aspect_term = process_content(sent_tokenizer)
    return aspect_term

# threshold value 
similarity_val_threshold = 0.6 
path = 'dataset/trained_data'
limit_threshold = 3 #1,2 for negative and 3,4 and 5 for positive
writeFileName = './trainData.json'

# Predefine aspect terms in the file and associated aspect terms considered in the project
aspects = {"Service":"service", "Rooms" : "room", "Cleanliness": "cleanliness",
                      "Value":"value",
                      "Location": "location", "Sleep Comfort":"sleep",
                      "Business service (e.g., Free WIFI)":"business","Reception":"reception"}

# storing and printing aspect terms
aspect_terms = aspects.keys()
aspect_term_modified=unique(aspects.values())
print("Aspect Terms: ")
print(aspect_terms)
print("aspect terms considered: ")
print(aspect_term_modified)


stops=stopwords.words("english")
stops.extend(['.', ',', "'s", "n't", '!', '(', ')', '-', ':', '!', '?', '...', '..', '+', ';', '<', '>'])
stop_words = set(stops)
stp_wrds=set(['pruce','roon','sweet','unit','cheq','bathrom','chek', 'locaton'])
stop_words_modified=list(stop_words-w2v)
start_time = time.time()
listing = os.listdir(path)
aspect_terms_found = []



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
                    asp_word_found = extractterm(s)
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
