
import nltk
import pke

nltk.download('stopwords')


def getSummaries(doc, labels, n=3):
    """
    Added variable n number for multiple keyword extraction
    """
    summaries = []
    segment = ''
    numSent = 0
    # k = 0 #not using?
    for i, sent in enumerate(doc):
        ##looping through sentences
        if labels[i] == 1 and segment != '':
            ## label==1 if its the first sentence of a segment
            ##(but doesn't enter if it's the first sentence of the first segment)
            extractor = pke.unsupervised.TopicRank()
            extractor.load_document(input=segment, language='en')
            extractor.candidate_selection()
            extractor.candidate_weighting()
            summaries.append(extractor.get_n_best(n))
            # k = k+1 #not using?
            segment = sent
            numSent = 1
        else:
            ##First sentence enters here
            ##+ then all sentences WITHIN segments after (i.e. not the first sents)
            segment = segment + " " + sent
            numSent = numSent + 1

    ##'segment' is a long string of sentences from the last segment
    ## the next few lines deal with the final segment (i think..)
    extractor = pke.unsupervised.TopicRank()
    extractor.load_document(input=segment, language='en')
    extractor.candidate_selection()
    extractor.candidate_weighting()

    summaries.append(extractor.get_n_best(n))

    return summaries

extractor = pke.unsupervised.TopicRank()
extractor.load_document(input=segment, language='en')
extractor.candidate_selection()
extractor.candidate_weighting()