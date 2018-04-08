import time
import os
import json
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import wordnet
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from matplotlib.cbook import unique


nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

ps = PorterStemmer()
lemmatizer=WordNetLemmatizer()
tokenizer_reg = RegexpTokenizer(r'\w+')
tokenizer = nltk.data.load("tokenizers/punkt/english.pickle")

similarity_threshold = 0.6
path = 'json/'
sentiment_threshold = 3 #(1 and 2 is negative, 3,4 and 5 positive)
writeFileName = './train_data.json' #trained json data file

def writeToFile(fileName, data_in):
    with open(fileName, 'w') as fp:
        json.dump(data_in, fp)

def aspect_term_extractor(review_line):
    review_line = review_line.lower()
    review_linewords = tokenizer_reg.tokenize(review_line)
    filtered_review_line = [wd for wd in review_linewords if not wd in stop_words_modified]
    tagged = nltk.pos_tag(filtered_review_line)
    max_similarity_value = 0
    asp_word_found = False
    adj_found = False
    for (w1, t1) in tagged:
        for wn in aspect_terms_found:
            if(wn=='Overall'):
                continue

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

        if(t1 in adjective):
            adj_found=True

    if(asp_word_found):
        return asp_word
    elif(adj_found):
        return 'Overall'
    else:
        asp_word = 'Null'
        return asp_word



defined_aspects = {"Rooms" : "room", "Cleanliness": "cleanliness",
                      "Value":"value", "Service":"service",
                      "Location": "location", "Sleep Quality":"sleep",
                      "Business service (e.g., internet access)":"internet", "Business service":"wireless","Check in / front desk":"reception"}  #Business service (e.g., internet access)', 'Check in / front desk'

noun = ["NN", "NNS", "NNP", "NNPS"]
adjective = ["JJ", "JJR", "JJS"]

stops=stopwords.words("english")
stops.extend(['.', ',', "'s", "n't", '!', '(', ')', '-', ':', '!', '?', '...', '..', '+', ';', '<', '>'])
stop_words = set(stops)
a = set(['wasn', 'mustn', 'didnt', 'didn', 'shouldn', 'mightn','weren', 'not','nor','no','aren','haven','isn','doesn', 'hasn'])
stop_words_modified=list(stop_words-a)
start_time = time.time()


defined_aspect_terms = defined_aspects.keys()
def_aspect_term_modified=unique(defined_aspects.values())

start_time = time.time()
listing = os.listdir(path)

aspect_terms_found = []

data_list=[]
f=0
for infile in listing:
    current_file = path+infile
    f+=1
    print(f)
    if f < 2 :
        with open(current_file) as data_file:
            data = json.load(data_file)
            reviews = data['Reviews']
            for rev in reviews:
                content = rev['Content']
                rating = rev['Ratings']
                aspect_terms_found = [ww for ww in rating]
                sentences = tokenizer.tokenize(content)
                for s in sentences:
                    asp_word_found = aspect_term_extractor(s)
                    data={}
                    if(asp_word_found!='Null' and asp_word_found!='Overall' and asp_word_found in aspect_terms_found):
                        label=rating[asp_word_found]
                        data['text']=s
                        data['label'] = 'pos' if int(label) > sentiment_threshold else 'neg'
                        data_list.append(data)
                    elif (asp_word_found == 'Overall' and 'Overall' in aspect_terms_found):
                        label = rating[asp_word_found]
                        data['text'] = s
                        data['label'] = 'pos' if float(label) > sentiment_threshold else 'neg'
                        data_list.append(data)


writeToFile(writeFileName,data_list)


