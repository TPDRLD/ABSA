import xml.etree.ElementTree as ET
from libraries.baselines import Corpus

from stanford_corenlp_python import jsonrpc
from simplejson import loads


def trip_advisor_18():
    corpora = dict()
    corpora['hotels'] = dict()
    train_filename = 'datasets/Trip_advisor_train_final.xml'
    trial_filename = 'datasets/Trip_advisor_test_final.xml'

    reviews = ET.parse(train_filename).getroot().findall('Review') + \
              ET.parse(trial_filename).getroot().findall('Review')

    sentences = []
    for r in reviews:
        sentences += r.find('sentences').getchildren()

    corpus = Corpus(sentences)
    corpus.size()


def main():
 
    server = jsonrpc.ServerProxy(jsonrpc.JsonRpc20(),
                                 jsonrpc.TransportTcpIp(addr=("127.0.0.1",
                                                              8080)))

    result = loads(server.parse("Hello world.  It is so beautiful"))
    print "Result", result

    corpora = trip_advisor_18()
    train_hotels = corpora['hotels']['trainset']['corpus']

    for s in train_hotels.corpus:
        print s.text

    

if __name__ == '__main__':
    main()
