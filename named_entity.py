import spacy
from spacy.matcher import PhraseMatcher
from pathlib import Path
import random
import csv
import numpy
from spacy.util import minibatch, compounding
import re 
import json

#convert the string index
def offseter(lbl, doc, matchitem):
    o_one = len(str(doc[0:matchitem[1]]))
    subdoc = doc[matchitem[1]:matchitem[2]]
    o_two = o_one + len(str(subdoc))
    return (o_one, o_two, lbl)

#load sapcy model
nlp = spacy.load('en_core_web_sm')

if 'ner' not in nlp.pipe_names:
    ner = nlp.create_pipe('ner')
    nlp.add_pipe('ner', Last=True)
else:
    ner = nlp.get_pipe('ner')

#new label to insert
label = "CITATION"
ner.add_label('CITATION')

matcher = PhraseMatcher(nlp.vocab)
def on_match(matcher, doc, id, matches):
      print('Matched!', matches)

   
#Gather training data
to_train_ents = []
res = []
with open('text.txt', 'r') as doc:
    line = True   
    while line:
        line = doc.readline()
        #use of regular expression to find all the citations with the following pattern
        citation_pattern = re.findall(r'\[\d+\]\s\w+\s\d+\s\(\w+\)|\[\d+\]\s\w+\s\d+', line, re.M|re.I)
        if(citation_pattern):
            citation = citation_pattern
            matcher.add(label, on_match, nlp(citation[0]))
            mnlp_line = nlp(line)    
            matches = matcher(mnlp_line)
            res = [offseter(label, mnlp_line, x) for x in matches]
            to_train_ents.append((citation[0], dict(entities=res)))


#use only named entity recognizer
other_pipes = [pipe
    for pipe
    in nlp.pipe_names
    if pipe!= 'ner']

feeds = []
nlp.begin_training()
for itn in range(20):
    random.shuffle(to_train_ents)
    losses = {}

    # batch up the examples using spaCy's minibatch
    batches = minibatch(to_train_ents, size=compounding(4.0, 32.0, 1.001))
    for batch in batches:
        texts, annotations = zip(*batch)
        nlp.update(texts, annotations, drop=0.5, losses=losses)
        print([batch[1]])
        #save results to json file
    with open('citations.json', mode='w', encoding='utf-8') as f:
        json.dump([],f)
    with open('citations.json', mode='w', encoding='utf-8') as json_file:
        feeds.append([batch[1]])
        json.dump(feeds, json_file)

#save model to disk
nlp.to_disk("/model", disable=['tokenizer'])