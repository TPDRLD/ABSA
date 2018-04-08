Other Tools required:
-- Stanford CoreNLP  
-- NLTK (WordNet)

Other Files required:
-- json files of trip advisor dataset - for run time purpose , uploading only 4 json files, so that code can be easily run to check
-- reviews.xml file containing the data from the trip advisor dataset. converted from json to xml
-- a few other json files are required, they have been uploaded after running the code for minimal data, to show as sample, the intermediate and output processes

This repository contains all the codes required for Aspect Based Sentiment Analysis,  
1. Run the get_data.py to get reviews from XML file inside <text></text> tag.
2. Run corenlp.py, The file here has been edited as per need, the original comes from Stanford CoreNLP Parser, to run json rpc server.
4. Run client.py to access json client
5. Run aspect_term_extraction.py  to extract the aspect terms.
6. Run aspect_term_polarity.py to get polarity and naive bayes results of the aspect terms extracted previously.
7. Run polarity_rating_analyzer.py to get the final output along with polarity rating.
