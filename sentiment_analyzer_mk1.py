from preprocessor_mk1 import TextPreprocessor
from textblob import TextBlob
from sklearn.decomposition import TruncatedSVD
from collections import defaultdict, Counter
from topic_classifier import pick_classifier
import scipy.cluster.hierarchy as hac
import pickle
import pymongo
import random
import numpy as np


class TextSentimentAnalysis(object):

    def __init__(self, classifier_filepath, sentiment_lexicon_path, processor,
                 db_name, coll_name, uri):

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

    def _launch_mongo(self, db_name, coll_name, uri=None):
        mc = pymongo.MongoClient(uri)
        db = mc[db_name]
        coll = db[coll_name]
        return coll

    def _create_blob(self, article):
        blob = TextBlob(article, classifier=self.sentiment_classifier)
        return blob

    def _simple_sentiment(self, word):
        sentiment = self.sentiment_lexicon.get(word, 0)
        return sentiment

    def _return_top_words(self, doc_top_dist, feature_names, n_top_words=50):
        index = np.argpartition(doc_top_dist, -3)[-3:]
        topic_dict = {}
        for top_num, ind in enumerate(index):
            topic = self.processor.lda_model.components_[ind]
            topic_top_n_words = set([feature_names[i] for i in
                                    topic.argsort()[:-n_top_words - 1:-1]])
            topic_dict[top_num] = topic_top_n_words

        return topic_dict

    def _lda_dim_reduction(self, vectorized_tokens, vectorizer):
        doc_top_dist = self.processor.lda_model.transform(vectorized_tokens)
        doc_top_dist = doc_top_dist[0]
        feature_names = vectorizer.get_feature_names()
        topic_dict = self._return_top_words(doc_top_dist, feature_names)

        return topic_dict

    def _sentiment_per_sentence(self, topic_dict, blob):
        sentiments_dict = defaultdict(lambda: defaultdict(list))
        for k, v in topic_dict.iteritems():
            predictions = Counter()
            topic_sentences = []
            sentence_sentiment = []
            for sentence in blob.sentences:
                sent_set = set(self.processor._tokenize(sentence))
                if len(v.intersection(sent_set)) > 1:  # arbitrary number
                    sent_set = [sent_set]
                    sent_vect = self.processor._vectorize(sent_set)
                    prediction = self.sentiment_classifier.predict(sent_vect)[0]
                    topic_sentences.append(str(sentence))
                    sentence_sentiment.append(prediction)
                    predictions[prediction] += 1
            if len(predictions) < 1:
                topic_pred = 1
            else:
                topic_pred = predictions.most_common(1)[0][0]
            sentiments_dict['topic_{}'.format(k)]['sentences'] = topic_sentences
            sentiments_dict['topic_{}'.format(k)]['sentence_sentiment'] = sentence_sentiment
            sentiments_dict['topic_{}'.format(k)]['topic_sentiment'] = topic_pred

        return sentiments_dict

    def _find_article_sentiment(self, article, vectorized_tokens, vectorizer):
        blob = self._create_blob(article)
        topic_dict = self._lda_dim_reduction(vectorized_tokens, vectorizer)
        sentiments_dict = self._sentiment_per_sentence(topic_dict, blob)

        return sentiments_dict

    def _matrix_svd(self, matrix):
        skl_u = self.tsvd.transform(matrix)
        skl_u = skl_u[:, :self.tsvd_cut]
        return skl_u

    def _train_trunc_svd(self, matrix):
        self.tsvd = TruncatedSVD(n_components=100, n_iter=50).fit(matrix)
        for i in xrange(len(self.tsvd.explained_variance_ratio_)):
            if self.tsvd.explained_variance_ratio_[:i].sum() > self.exp_var_desired:
                print i, self.tsvd.explained_variance_ratio_[:i].sum()
                self.tsvd_cut = i
        with open('svd_model.pkl', 'wb') as f:
            pickle.dump(self.tsvd, f)

    def _train_clusters(self, topics_list):
        vectorized = self.processor._vectorize(topics_list).toarray()
        vectorized = np.nan_to_num(vectorized)
        self._train_trunc_svd(vectorized)
        skl_u = self._matrix_svd(vectorized)
        # dist = 1 - cosine_similarity(skl_u)
        link_matrix = hac.linkage(skl_u, metric='cosine', method='average')
        h_cluster = hac.fcluster(link_matrix, t=0.05, criterion='distance')

        self.n_h_clusters = len(np.unique(np.array(h_cluster)))

        return vectorized, skl_u, h_cluster

    def _select_best_classifier(self, X_reduced, y):
        best_estimator, best_params = pick_classifier(X_reduced, y)
        self.cluster_classifier = best_estimator
        self.cluster_classifier.set_params(**best_params)
        self.cluster_classifier.fit(X_reduced, y)

    # --------- All private methods above this line -------

    def extract_topic_features(self):
        count, error_count = 0, 0
        print "Using LDA to Extract Topic Features"
        for doc in self.coll.find(snapshot=True).batch_size(25):
            try:
                doc_id = doc['_id']
                article = doc['article']
                cleaned = self.processor._correct_sentences(article)
                vectorizer, vectorized_tokens = self.processor.generate_vectors(cleaned)
                topic_dict = self._lda_dim_reduction(vectorized_tokens, vectorizer)
                for k, v in topic_dict.iteritems():
                    self.coll.find_one_and_update({'_id': doc_id}, {'$set': {'topic_{}'.format(k): {'topic_features': list(v)}}})
                count += 1
                print 'Success # {}'.format(count)
            except TypeError:
                error_count += 1
                print 'TypeError, Moving on. Error # {}'.format(error_count)
            except ValueError:
                error_count += 1
                print 'ValueError, Moving on. Error # {}'.format(error_count)

    def cluster_by_topic_similarity(self, desired_exp_var=0.9):
        self.exp_var_desired = desired_exp_var
        count, error_count = 0, 0
        topics_list = []
        article_ids = []
        article_topics = []
        for doc in self.coll.find(snapshot=True).batch_size(25):
            try:
                doc_id = doc['_id']
                topics = ['topic_0', 'topic_1', 'topic_2']
                for topic in topics:
                    topics_list.append(doc[topic]['topic_features'])
                    article_ids.append(doc_id)
                    article_topics.append(topic)
                count += 1
                print "Pass #{}".format(count)
            except:
                error_count += 1
                print "ERROR, MOVING ON #{}".format(error_count)

        vectorized, u, h_cluster = self._train_clusters(topics_list)

        for index, doc_id in enumerate(article_ids):
            topic = article_topics[index]
            label = h_cluster[index].astype(np.int64)
            self.coll.find_one_and_update({'_id': doc_id}, {'$set': {topic + '.label': label}})

        self._select_best_classifier(u, h_cluster)

        return article_ids, article_topics, u, h_cluster

    def corpus_analytics(self):
        count, error_count = 0, 0
        print 'Analyzing Articles and Storing in Mongo'
        for doc in self.coll.find(snapshot=True).batch_size(25):
            try:
                doc_id = doc['_id']
                article = doc['article']
                cleaned = self.processor._correct_sentences(article)
                vectorizer, vectorized_tokens = self.processor.generate_vectors(cleaned)
                sentiments_dict = self._find_article_sentiment(cleaned, vectorized_tokens, vectorizer)
                for k, v in sentiments_dict.iteritems():
                    self.coll.find_one_and_update({'_id': doc_id},
                                                  {'$set': {k + '.sentences': v['sentences'],
                                                            k + '.sentence_sentiment': v['sentence_sentiment'],
                                                            k + '.topic_sentiment': v['topic_sentiment']}})
                count += 1
                print 'Pass #{}'.format(count)
            except TypeError:
                error_count += 1
                print 'ERROR, MOVING ON #{}'.format(error_count)
            except ValueError:
                error_count += 1
                print 'ValueError, Moving On #{}'.format(error_count)
        print 'COMPLETE'

    def classify_new_article(self, url):
        article = self.processor.new_article(url)
        vectorizer, vectorized = self.processor.generate_vectors(article)
        topic_dict = self._lda_dim_reduction(vectorized, vectorizer)
        sentiments_dict = self._find_article_sentiment(article,
                                                       vectorized, vectorizer)
        try:
            self.coll.insert_one({'article': article, 'url': url,
                                  'topic_0': list(topic_dict[0]),
                                  'topic_1': list(topic_dict[1]),
                                  'topic_2': list(topic_dict[2])})
        except pymongo.errors.DuplicateKeyError:
            print 'this url already exists I do not need to do anything'

        topics_list = []
        for k, v in topic_dict.iteritems():
            topics_list.append(list(v))
        vectorized = self.processor._vectorize(topics_list).toarray()
        u = self._matrix_svd(vectorized)
        class_predict = self.cluster_classifier.predict(u)

        return class_predict, sentiments_dict

    def report_for_article(self, class_predict, sentiments_dict, article_ids,
                           article_topics, h_cluster):

        article_dict = defaultdict(list)
        for i, classification in enumerate(class_predict):
            index = np.argwhere(np.array(h_cluster) == classification)
            for ind in index:
                doc_id = article_ids[ind[0]]
                topic = article_topics[ind[0]]
                document = self.coll.find_one({'_id': doc_id})
                try:
                    doc_sentiments = document[topic]['topic_sentiment']
                except KeyError:
                    continue
                sentiment_score = doc_sentiments
                new_article_score = sentiments_dict['topic{}'.format(i)]['topic_sentiment']
                if new_article_score != sentiment_score:
                    article_dict[i].append(document['url'])

        return article_dict


if __name__ == '__main__':
    with open('local_access.txt', 'r') as f:
        access_tokens = []
        for line in f:
            line = line.strip()
            access_tokens.append(line)
    db_name = access_tokens[1]
    coll_name = access_tokens[2]
    uri = 'mongodb://root:{}@localhost'.format(access_tokens[0])
    processor_filepath = '/home/bitnami/processor.pkl'
    lda_model_filepath = '/home/bitnami/lda_model.pkl'
    classifier_filepath = '/home/bitnami/sentiment_classifier.pkl'
    lexicon_filepath = '/home/bitnami/sentiment_lexicon.pkl'
    prep = TextPreprocessor(lemmatize = True, vectorizer = processor_filepath, lda_model = lda_model_filepath)
    sentiment_analyzer = TextSentimentAnalysis(classifier_filepath, lexicon_filepath, prep, db_name, coll_name, uri)
    # sentiment_analyzer.extract_topic_features()
    result = sentiment_analyzer.cluster_by_topic_similarity()
    article_ids, article_topics, svd_matrix, h_cluster = result
    print np.unique(np.array(h_cluster), return_counts = True)
    sentiment_analyzer.corpus_analytics()
    # url = 'http://www.cnn.com/2017/10/29/politics/angus-king-collusion-calls-sotu/index.html'
    url = 'http://www.cnn.com/2017/11/01/politics/trump-justice-laughing-stock/index.html'
    class_predict, sentiments_dict = sentiment_analyzer.classify_new_article(url)
    article_dict = sentiment_analyzer.report_for_article(class_predict,
                                                         sentiments_dict, article_ids, article_topics, h_cluster)
    print random.choice(article_dict[0])
    print random.choice(article_dict[1])
    print random.choice(article_dict[2])
