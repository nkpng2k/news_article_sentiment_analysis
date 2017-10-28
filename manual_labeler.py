from textblob import TextBlob
from preprocessor_mk1 import TextPreprocessor
import pymongo
import random

class Jarvis(object):

    def __init__(self, preprocessor, db_name, coll_name, uri = None):
        self.processor = preprocessor
        self.coll = self._connect_to_database(db_name, coll_name, uri)
        self.labels_coll = self._connect_to_database(db_name, 'sentiment_labels', uri)
        self.doc_list = []

    def _connect_to_database(self, db_name, coll_name, uri = None):
        mc = pymongo.MongoClient(uri)
        db = mc[db_name]
        coll = db[coll_name]
        return coll

    def jarvis_tell_me(self):
        sentences_to_label = []
        count, error_count = 0, 0
        for doc in self.coll.find().batch_size(25):
            try:
                article = doc['article']
                cleaned = self.processor._correct_sentences(article)
                blob = TextBlob(cleaned)
                num_sentences = int(len(blob.sentences) * 0.25)
                blob_sentences = random.sample(blob.sentences, num_sentences)
                sentences_to_label.extend(blob_sentences)
                count += 1
                print 'Pass: {}'.format(count)
            except TypeError:
                error_count += 1
                print 'Error # {}. TypeError: '.format(error_count), type(article)
        print len(sentences_to_label)
        return sentences_to_label

    def jarvis_label(self, sentences_to_label):
        for sentence in sentences_to_label:
            print sentence
            label = raw_input('What is the sentiment of this sentence: ')
            string_sentence = str(sentence)
            if label == 'skip':
                continue
            if label == 'jarvis lets stop':
                break
            else:
                self.labels_coll.insert_one({'sentence':string_sentence, 'label':label})



if __name__ == '__main__':
    with open('local_access.txt','r') as f:
        access_tokens = []
        for line in f:
            line = line.strip()
            access_tokens.append(line)
    db_name = access_tokens[1]
    coll_name = access_tokens[2]
    uri = 'mongodb://root:{}@localhost'.format(access_tokens[0])
    processor_filepath = '/home/bitnami/processor.pkl'
    classifier_filepath = '/home/bitnami/naivebayesclassifier.pkl'
    lda_model_filepath = '/home/bitnami/lda_model.pkl'
    prep = TextPreprocessor(lemmatize = True, vectorizer = processor_filepath, lda_model = lda_model_filepath)
    my_jarvis = Jarvis(prep, db_name, coll_name, uri)
    sentences_to_label = my_jarvis.jarvis_tell_me()
    my_jarvis.jarvis_label(sentences_to_label)
