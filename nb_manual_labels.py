from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
import preprocessor_mk1
import sentiment_analyzer_mk1
import pymongo
import pickle

class TrainClassifier(object):

    def __init__(self, preprocessor, svd_reducer, db_name, coll_name, uri):
        self.processor = preprocessor
        with open(svd_reducer, 'rb') as f:
            self.svd_model = pickle.load(f)
        self.coll = self._launch_mongo(db_name, coll_name, uri)

    def _launch_mongo(self, db_name, coll_name, uri = None):
        mc = pymongo.MongoClient(uri)
        db = mc[db_name]
        coll = db[coll_name]
        return coll

    def read_in_data(self):
        strings = []
        labels = []
        for doc in self.coll.find():
            sentence = doc['sentence']
            label = doc['label']
            strings.append(sentence)
            labels.append(label)

        return strings, labels

    def grid_search(self, models, parameters, strings, labels):
        tokens = preprocessor._tokenize(strings)
        vectorized = preprocessor._vectorize(tokens)
        X_train, X_test, y_train, y_test = train_test_split(vectorized, labels, test_size = 0.2, shuffle = True)

            for i, estimator in enumerate(estimators_list):
            print 'Grid Search Loop'
            best_score = 0
            best_params = None
            best_estimator = None
            clf = GridSearchCV(estimator, params_list[i], cv = 5, scoring = 'accuracy', verbose = 2)
            clf.fit(X_train, y_train)
            clf_score = clf.score(X_test, y_test)
            if clf_score > best_score:
                best_score = clf_score
                best_params = clf.best_params_
                best_estimator = estimator

            return best_estimator, best_params, best_score, vectorized

    def pickle_classifier(self, best_estimator, best_params, X, y):
        classifier = best_estimator
        classifier.set_params(**best_params)
        classifier.fit(X, y)

        with open('sentiment_classifier.pkl', 'wb') as f:
            pickle.dump(classifier, f)

        print 'Pickle-ing Complete'

if __name__ == "__main__":
    processor_filepath = '/home/bitnami/processor.pkl'
    lda_model_filepath = '/home/bitnami/lda_model.pkl'
    classifier_filepath = '/home/bitnami/naivebayesclassifier.pkl'
    lexicon_filepath = '/home/bitnami/sentiment_lexicon.pkl'
    svd_reducer = '/home/bitnami/svd_model.pkl'

    rand_forest = RandomForestClassifier()
    rand_forest_params = {'n_estimators': [10,100,1000], 'max_features': [0.1, 0.2, 0.5, 0.8], 'min_samples_split': [2, 4, 8]}
    ada_boost = AdaBoostClassifier()
    ada_boost_params = {'n_estimators':[10, 50, 100, 150, 250, 500], 'learning_rate': [0.001, 0.01, 0.1]}
    gaussian_nb = GaussianNB()
    gaussian_params = {}
    multi_nb = MultinomialNB()
    multi_params = {'alpha': [0, 0.25, 0.5, 0.75, 1]}

    models = [rand_forest, ada_boost, gaussian_nb, multi_nb]
    parameters = [rand_forest_params, ada_boost_params, gaussian_params, multi_params]

    prep = TextPreprocessor(lemmatize = True, vectorizer = processor_filepath, lda_model = lda_model_filepath)
    classifier_manual = TrainClassifier(preprocessor = prep, svd_reducer = svd_reducer, db_name = db_name,
                                        coll_name = coll_name, uri = uri)
    strings, labels = classifier_manual.read_in_data()
    results = classifier_manual.grid_search(models, parameters, strings, labels)
    best_estimator, best_params, best_score, vectorized = results
    classifier_manual.pickle_classifier(best_estimator, best_params, vectorized, labels)
