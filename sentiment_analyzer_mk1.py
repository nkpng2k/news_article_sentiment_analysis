from preprocessor_mk1 import TextPreprocessor
from textblob import TextBlob
from textblob.classifiers import NaiveBayesClassifier
from sklearn.decomposition import LatentDirichletAllocation
from collections import defaultdict
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity
import outside_functions as of
import scipy.cluster.hierarchy as hac
import matplotlib.pyplot as plt
import pickle
import pymongo
import numpy as np

class TextSentimentAnalysis(object):

    def __init__(self, classifier_filepath, processor):
        with open(classifier_filepath, 'rb') as f:
            self.classifier = pickle.load(f)
        self.processor = processor
        self.n_h_clusters = None
        self.n_dbscan_cluster = None
        self.power_level = None
        print 'all dependencies loaded'

    def _launch_mongo(self, db_name, coll_name, uri = None):
        mc = pymongo.MongoClient(uri)
        db = mc[db_name]
        coll = db[coll_name]
        return coll

    def _create_blob(self, article):
        blob = TextBlob(article, classifier = self.classifier)
        return blob

    def _return_top_words(self, model, feature_names, n_top_words = 50):
        topic_dict = {}
        for topic_idx, topic in enumerate(model.components_):
            topic_top_n_words = set([feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]])
            topic_dict[topic_idx] = topic_top_n_words

        return topic_dict

    def _whole_doc_sentiment(self, article):
        article_prob_dist = self.classifier.prob_classify(article)
        prediction = article_prob_dist.max()
        pred_prob = article_prob_dist.prob(prediction)

        return prediction, pred_prob

    def _sentiment_per_sentence(self, topic_dict, blob):
        sentiments_dict = defaultdict(lambda : defaultdict(list))
        for k, v in topic_dict.iteritems():
            for sentence in blob.sentences:
                sent_set = set(sentence.split())
                if len(v.intersection(sent_set)) > 2: #2 is an arbitrary number
                    sent_dist = self.classifier.prob_classify(sentence)
                    sent_pred = sent_dist.max()
                    sentiments_dict['topic_{}'.format(k)]['sentences'].append(str(sentence))
                    sentiments_dict['topic_{}'.format(k)]['predictions'].append(sent_pred)
            sentiments_dict['topic_{}'.format(k)]['topic_features'] = list(v)

        return sentiments_dict

    def _lda_dim_reduction(self, vectorized_tokens, vectorizer):
        lda = LatentDirichletAllocation(n_components = 3, learning_method = 'batch').fit(vectorized_tokens)
        feature_names = vectorizer.get_feature_names()
        topic_dict = self._return_top_words(lda, feature_names)

        return topic_dict

    def _matrix_svd(self, matrix):
        if self.power_level>1:
            print "No. Power level can't be more than 1"
            self.power_level=0.9
        #decompose matrix into U,S,V components
        U, Sigma, VT = np.linalg.svd(matrix)
        #shrink matrix to latent features that account for power_level fraction of total power
        power = Sigma**2
        passed_thresh = np.argwhere(np.cumsum(power)/np.sum(power) >= self.power_level)
        U_trunc = U[:, :passed_thresh[0][0]] #rows: articles, columns: latent features
        Sigma_trunc = Sigma[:passed_thresh[0][0]]
        VT_trunc = VT[:, :passed_thresh[0][0]] #rows: features, columns: latent articles
        return U_trunc, Sigma_trunc, VT_trunc

    def _find_article_sentiment(self, article, vectorized_tokens, vectorizer):
        blob = self._create_blob(article)
        art_pred, art_prob = self._whole_doc_sentiment(article)
        topic_dict = self._lda_dim_reduction(vectorized_tokens, vectorizer)
        sentiments_dict = self._sentiment_per_sentence(topic_dict, blob)

        return art_pred, sentiments_dict

    def _train_clusters(self, topics_list):
        vectorized = self.processor._vectorize(topics_list).toarray()
        u, sigma, vt = self._matrix_svd(vectorized)
        link_matrix = hac.linkage(u, metric = 'cosine', method = 'average')
        h_cluster = hac.fcluster(link_matrix, t = 0.05, criterion = 'distance')

        dbscan = DBSCAN(min_samples = 5, metric = 'cosine', n_jobs = -1).fit(u)

        self.n_h_clusters = len(np.unique(np.array(h_cluster)))
        self.n_dbscan_cluster = len(np.unique(dbscan.labels_))

        return u, h_cluster, dbscan

    # --------- All private methods above this line -------

    def corpus_analytics(self, db_name, coll_name, uri = None):
        coll = self._launch_mongo(db_name, coll_name, uri)
        count, error_count = 0, 0
        print 'Analyzing Articles and Storing in Mongo'
        for doc in coll.find(snapshot = True).batch_size(25):
            try:
                doc_id = doc['_id']
                article = doc['article']
                cleaned = self.processor._correct_sentences(article)
                vectorizer, vectorized_tokens = self.processor.generate_vectors(cleaned)
                art_pred, sentiments_dict = self._find_article_sentiment(cleaned, vectorized_tokens, vectorizer)
                coll.find_one_and_update({'_id':doc_id}, {'$set':{'sentiment':sentiments_dict}})
                count += 1
                print 'Pass #{}'.format(count)
            except TypeError:
                error_count += 1
                print 'ERROR, MOVING ON #{}'.format(error_count)
            except ValueError:
                error_count += 1
                print 'ValueError, Moving On #{}'.format(error_count)
        print 'COMPLETE'

    #--------- TODO: NEED TO COMPLETE FUNCTIONS BELOW THIS LINE ------------

    def cluster_by_topic_similarity(self, db_name, coll_name, uri = None, power_level = 0.9):
        coll = self._launch_mongo(db_name, coll_name, uri)
        self.power_level = power_level
        count, error_count = 0, 0
        topics_list = []
        article_ids = []
        for doc in coll.find(snapshot = True).batch_size(25).limit(5000):
            try:
                doc_id = doc['_id']
                for k, v in doc['sentiment'].iteritems():
                    topics_list.append(v['topic_features'])
                    article_ids.append((doc_id, k))
                count += 1
                print "Pass #{}".format(count)
            except:
                error_count += 1
                print "ERROR, MOVING ON #{}".format(error_count)

        u, h_cluster, dbscan = self._train_clusters(topics_list)

        return article_ids, u, h_cluster, dbscan

    def predict_on_new_article(self, url):
        article = self.processor.new_article(url)
        vectorizer, vectorized = self.processor.generate_vectors(article)
        art_pred, sentiments_dict = self._find_article_sentiment(article, vectorized, vectorizer)
        u, sigma, vt = self._matrix_svd(vectorized)
        prediction = dbscan.fit_predict(u)

        return prediction

if __name__ == '__main__':
    db_name = 'test_articles'
    coll_name = 'article_text_data'
    uri = 'mongodb://root:9EThDhBJiBGP@localhost'
    processor_filepath = '/home/bitnami/processor.pkl'
    classifier_filepath = '/home/bitnami/naivebayesclassifier.pkl'
    prep = TextPreprocessor(vectorizer = processor_filepath)
    sentiment_analyzer = TextSentimentAnalysis(classifier_filepath, prep)
    # sentiment_analyzer.corpus_analytics(db_name, coll_name, uri)
    result = sentiment_analyzer.cluster_by_topic_similarity(db_name, coll_name, uri, 0.9)
    article_ids, svd_matrix, h_cluster, dbscan = result
    print np.unique(np.array(h_cluster), return_counts = True)
    print np.unique(dbscan.labels_, return_counts = True)
    url = ''
    prediction = sentiment_analyzer.predict_on_new_article(url)
