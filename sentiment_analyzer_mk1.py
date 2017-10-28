from preprocessor_mk1 import TextPreprocessor
from textblob import TextBlob
from textblob.classifiers import NaiveBayesClassifier
from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity
from topic_classifier import pick_classifier
import outside_functions as of
import scipy.cluster.hierarchy as hac
import matplotlib.pyplot as plt
import pickle
import pymongo
import numpy as np

class TextSentimentAnalysis(object):

    def __init__(self, classifier_filepath, sentiment_lexicon_path, processor, db_name, coll_name, uri):
        self.coll = self._launch_mongo(db_name, coll_name, uri)
        with open(classifier_filepath, 'rb') as f:
            self.sentiment_classifier = pickle.load(f)
        with open(sentiment_lexicon_path, 'rb') as f:
            self.sentiment_lexicon = pickle.load(f)
        self.processor = processor
        self.tsvd = None
        self.tsvd_cut = None
        self.n_h_clusters = None
        self.exp_var_desired = None
        self.cluster_classifier = None
        print 'all dependencies loaded'

    def _launch_mongo(self, db_name, coll_name, uri = None):
        mc = pymongo.MongoClient(uri)
        db = mc[db_name]
        coll = db[coll_name]
        return coll

    def _create_blob(self, article):
        blob = TextBlob(article, classifier = self.sentiment_classifier)
        return blob

    def _simple_sentiment(self, word):
        sentiment = self.sentiment_lexicon.get(word, 0)
        return sentiment

    def _return_top_words(self, doc_top_dist, feature_names, n_top_words = 50):
        index = np.argpartition(doc_top_dist, -3)[-3:]
        topic_dict = {}
        for top_num, ind in enumerate(index):
            topic = self.processor.lda_model.components_[ind]
            topic_top_n_words = set([feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]])
            topic_dict[top_num] = topic_top_n_words

        return topic_dict

    def _lda_dim_reduction(self, vectorized_tokens, vectorizer):
        doc_top_dist = self.processor.lda_model.transform(vectorized_tokens)
        doc_top_dist = doc_top_dist[0]
        feature_names = vectorizer.get_feature_names()
        topic_dict = self._return_top_words(doc_top_dist, feature_names)

        return topic_dict

    def _whole_doc_sentiment(self, article):
        article_prob_dist = self.sentiment_classifier.prob_classify(article)
        prediction = article_prob_dist.max()
        pred_prob = article_prob_dist.prob(prediction)

        return prediction, pred_prob

    def _sentiment_per_sentence(self, topic_dict, blob):
        sentiments_dict = defaultdict(lambda : defaultdict(list))
        for k, v in topic_dict.iteritems():
            for sentence in blob.sentences:
                sent_set = set(self.processor._tokenize(sentence))
                if len(v.intersection(sent_set)) >= 1: #2 is an arbitrary number
                    # sent_dist = self.sentiment_classifier.prob_classify(sentence)
                    # sent_pred = sent_dist.max()
                    sent_pred = 0
                    for word in sent_set:
                        sent_pred += self._simple_sentiment(word)
                    print sent_pred
                    sentiments_dict['topic_{}'.format(k)]['sentences'].append(str(sentence))
                    sentiments_dict['topic_{}'.format(k)]['predictions'].append(sent_pred)
            sentiments_dict['topic_{}'.format(k)]['topic_features'] = list(v)

        return sentiments_dict

    def _find_article_sentiment(self, article, vectorized_tokens, vectorizer):
        blob = self._create_blob(article)
        art_pred, art_prob = self._whole_doc_sentiment(article)
        topic_dict = self._lda_dim_reduction(vectorized_tokens, vectorizer)
        sentiments_dict = self._sentiment_per_sentence(topic_dict, blob)

        return art_pred, sentiments_dict

    def _matrix_svd(self, matrix):
        skl_u = self.tsvd.transform(matrix)
        skl_u = skl_u[:, :self.tsvd_cut]
        return skl_u

    def _train_trunc_svd(self, matrix):
        self.tsvd = TruncatedSVD(n_components = 100, n_iter = 50).fit(matrix)
        for i in xrange(len(self.tsvd.explained_variance_ratio_)):
            if self.tsvd.explained_variance_ratio_[:i].sum() > self.exp_var_desired:
                print i, self.tsvd.explained_variance_ratio_[:i].sum()
                return i

    def _train_clusters(self, topics_list):
        vectorized = self.processor._vectorize(topics_list).toarray()
        vectorized = np.nan_to_num(vectorized)
        self.tsvd_cut = self._train_trunc_svd(vectorized)
        skl_u = self._matrix_svd(vectorized)
        dist = 1 - cosine_similarity(skl_u)
        link_matrix = hac.linkage(dist, metric = 'cosine', method = 'average')
        h_cluster = hac.fcluster(link_matrix, t = 0.05, criterion = 'distance')

        self.n_h_clusters = len(np.unique(np.array(h_cluster)))

        return vectorized, skl_u, h_cluster

    def _select_best_classifier(self, X_reduced, y):
        best_estimator, best_params = pick_classifier(X_reduced, y)
        self.cluster_classifier = best_estimator
        self.cluster_classifier.set_params(**best_params)
        self.cluster_classifier.fit(X_reduced, y)

    # --------- All private methods above this line -------

    def corpus_analytics(self):
        count, error_count = 0, 0
        print 'Analyzing Articles and Storing in Mongo'
        for doc in self.coll.find(snapshot = True).batch_size(25):
            try:
                doc_id = doc['_id']
                article = doc['article']
                cleaned = self.processor._correct_sentences(article)
                vectorizer, vectorized_tokens = self.processor.generate_vectors(cleaned)
                art_pred, sentiments_dict = self._find_article_sentiment(cleaned, vectorized_tokens, vectorizer)
                self.coll.find_one_and_update({'_id':doc_id}, {'$set':{'sentiment':sentiments_dict}})
                count += 1
                print 'Pass #{}'.format(count)
            except TypeError:
                error_count += 1
                print 'ERROR, MOVING ON #{}'.format(error_count)
            except ValueError:
                error_count += 1
                print 'ValueError, Moving On #{}'.format(error_count)
        print 'COMPLETE'

    def cluster_by_topic_similarity(self, desired_exp_var = 0.9):
        self.exp_var_desired = desired_exp_var
        count, error_count = 0, 0
        topics_list = []
        article_ids = []
        article_topics = []
        for doc in self.coll.find(snapshot = True).batch_size(25).limit(1000):
            try:
                doc_id = doc['_id']
                for k, v in doc['sentiment'].iteritems():
                    topics_list.append(v['topic_features'])
                    article_ids.append(doc_id)
                    article_topics.append(k)
                count += 1
                print "Pass #{}".format(count)
            except:
                error_count += 1
                print "ERROR, MOVING ON #{}".format(error_count)

        vectorized, u, h_cluster = self._train_clusters(topics_list)

        for index, doc_id in enumerate(article_ids):
            topic = article_topics[index]
            label = h_cluster[index].astype(np.int64)
            print 'sentiment.' + topic + '.label', label
            self.coll.find_one_and_update({'_id':doc_id}, {'$set':{'sentiment.'+ topic + '.label': label}})

        self._select_best_classifier(u, h_cluster)

        return article_ids, article_topics, u, h_cluster

    def classify_new_article(self, url):
        article = self.processor.new_article(url)
        vectorizer, vectorized = self.processor.generate_vectors(article)
        art_pred, sentiments_dict = self._find_article_sentiment(article, vectorized, vectorizer)
        try:
            self.coll.insert_one({'url':url, 'article':article, 'sentiment':sentiments_dict})
        except pymongo.errors.DuplicateKeyError:
            print 'this url already exists I do not need to do anything with it'
        topics_list = []
        for k, v in sentiments_dict.iteritems():
            topics_list.append(v['topic_features'])
        vectorized = self.processor._vectorize(topics_list).toarray()
        u = self._matrix_svd(vectorized)
        class_predict = self.cluster_classifier.predict(u)

        return class_predict, sentiments_dict

    #TODO: THIS NEEDS TO BE FIXED ASAP
    def report_for_article(self, class_predict, sentiments_dict, article_ids, article_topics, h_cluster):
        article_dict = defaultdict(list)
        for i, classification in enumerate(class_predict):
            index = np.argwhere(np.array(h_cluster) == classification)
            for ind in index:
                doc_id = article_ids[ind[0]]
                topic = article_topics[ind[0]]
                document = self.coll.find_one({'_id':doc_id})
                try:
                    doc_sentiments = document['sentiment'][topic]['predictions']
                    print document['sentiment'][topic]
                except KeyError:
                    continue
                sentiment_score = sum(doc_sentiments)
                new_article_score = sum(sentiments_dict['topic{}'.format(i)]['predictions'])
                if new_article_score * sentiment_score < 0:
                    article_dict[i].append(document['article'])

        return article_dict


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
    lda_model_filepath = '/home/bitnami/lda_model.pkl'
    classifier_filepath = '/home/bitnami/naivebayesclassifier.pkl'
    lexicon_filepath = '/home/bitnami/sentiment_lexicon.pkl'
    prep = TextPreprocessor(lemmatize = True, vectorizer = processor_filepath, lda_model = lda_model_filepath)
    sentiment_analyzer = TextSentimentAnalysis(classifier_filepath, lexicon_filepath, prep, db_name, coll_name, uri)
    sentiment_analyzer.corpus_analytics() #only needs to be run the first time
    result = sentiment_analyzer.cluster_by_topic_similarity()
    article_ids, article_topics, svd_matrix, h_cluster = result
    print np.unique(np.array(h_cluster), return_counts = True)
    url = 'http://www.foxnews.com/politics/2017/10/24/gop-sen-jeff-flake-says-wont-seek-re-election-in-2018.html'
    result = sentiment_analyzer.classify_new_article(url)
    prediction, sentiments = result
    print prediction
    article_dict = sentiment_analyzer.report_for_article(prediction, sentiments, article_ids, article_topics, h_cluster)
    print article_dict
