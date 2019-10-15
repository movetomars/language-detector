#!/usr/bin/env python

from optparse import OptionParser
from itertools import islice
import math
import os, logging, re
import collections

def take(n, iterable):
    "Return first n items of the iterable as a list"
    return list(islice(iterable, n))

def preprocess(line):
    ## get rid of the staff at the end of the line
    line = line.rstrip()
    ## lower case
    line = line.lower()
    ## remove everything except characters and white space
    line = re.sub("[^a-z ]", '', line)

    tokens = line.split()
    ## update this when working with trigrams
    ## you can also change the preprocessing (keep numbers, do not lower case, etc.)
    tokens = ['$'+token+'$' for token in tokens]
    
    return tokens



def create_model(path):
    ## This is just some Python magic ...
    ## unigrams will return 0 if the key doesn't exist
    unigrams = collections.defaultdict(int)
    ## and then you have to figure out what bigrams will return
    bigrams = collections.defaultdict(lambda: collections.defaultdict(int))


    f = open(path, 'r')
    ## You shouldn't visit a token more than once
    for l in f.readlines():
        tokens = preprocess(l)
        ########################################

        #Updating the unigrams dictionary with unigram (character) counts
        if len(tokens) == 0:
            continue
        for token in tokens:
            for char in token:

                if char not in unigrams:
                    unigrams[char] = 0

                unigrams[char] += 1

        #Updating the bigrams dictionary with bigram character counts
        for token in tokens:
            for i, char in enumerate(token, 0):
                next = i + 1

                if char not in bigrams:
                    bigrams[char][token[next]] = 0

                if next < len(token):
                    bigrams[char][token[next]] += 1
        ########################################
            pass

    ## After calculating the counts, calculate the smoothed log probabilities
    ###############################################
    probabilities = collections.defaultdict(lambda: collections.defaultdict(int))


    for i, item in enumerate(bigrams.items(), 0):
        org = item[0]
        bi = item[1].items()
        count_1 = unigrams.items()[i][1]

        for n in bi:
            count_2 = n[1]
            prob = (float(count_2+1)) / (float(count_1+26))
            n = n[0]
            probabilities[org][n] = math.log(prob)

    #u_items = take(3, probabilities.iteritems())
    #print(u_items)
    #Returning probability model
    return probabilities
    ###############################################

def predict(file, model_en, model_es):
    prediction = None

    #implement a helper method to calculate the log probabilities
    def calc_logprob(file, probabilities):

        final_prob = 0
        f = open(file, 'r')

        for l in f.readlines():
            tokens = preprocess(l)

            if len(tokens) == 0:
                continue

            # For each bigram, access and summate its individual probability
            for token in tokens:
                for i, char in enumerate(token, 0):
                    next = i + 1

                    if next < len(token):
                        partial = probabilities[char][token[next]]
                        final_prob += partial

        #return the final logarithmic probability for the entire document
        return final_prob

    #Calculate the probability of each document given each language model
    prob_es = calc_logprob(file, model_es)
    prob_en = calc_logprob(file, model_en)

    #And choose
    if prob_es > prob_en:
        prediction = "Spanish"
    else:
        prediction = "English"

    return prediction

def main(en_tr, es_tr, folder_te):
    ## DO NOT CHANGE THIS METHOD

    ## STEP 1: create a model for English with en_tr
    model_en = create_model(en_tr)

    ## STEP 2: create a model for Spanish with es_tr
    model_es = create_model(es_tr)

    ## STEP 3: loop through all the files in folder_te and print prediction
    folder = os.path.join(folder_te, "en")
    print "Prediction for English documents in test:"
    for f in os.listdir(folder):
        f_path =  os.path.join(folder, f)
        print "%s\t%s" % (f, predict(f_path, model_en, model_es))
    
    folder = os.path.join(folder_te, "es")
    print "\nPrediction for Spanish documents in test:"
    for f in os.listdir(folder):
        f_path =  os.path.join(folder, f)
        print "%s\t%s" % (f, predict(f_path, model_en, model_es))

if __name__ == "__main__":
    ## DO NOT CHANGE THIS CODE

    usage = "usage: %prog [options] EN_TR ES_TR FOLDER_TE"
    parser = OptionParser(usage=usage)

    parser.add_option("-d", "--debug", action="store_true",
                      help="turn on debug mode")

    (options, args) = parser.parse_args()
    if len(args) != 3:
        parser.error("Please provide required arguments")

    if options.debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.CRITICAL)

    main(args[0], args[1], args[2])
