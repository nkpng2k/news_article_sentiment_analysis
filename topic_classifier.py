from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


"""
Topic classifier:
Input: X --> vectorized, dense matrix (output of svd)
       y --> labels (output of hierarchical clustering algo)
Output: model
"""

def pick_classifier(X_reduced, y):
    rand_forest = RandomForestClassifier()
    rand_forest_params = {'n_estimators': [10,100,1000], 'max_features': [0.1, 0.2, 0.5, 0.8], 'min_samples_split': [2, 4, 8]}
    grad_boost = GradientBoostingClassifier()
    grad_boost_params = {'learning_rate': [0.001, 0.01, 0.1], 'max_depth': [1,2,3,5]}
    ada_boost = AdaBoostClassifier()
    ada_boost_params = {'n_estimators':[10, 50, 100, 150, 250, 500], 'learning_rate': [0.001, 0.01, 0.1]}

    estimators_list = [rand_forest, grad_boost, ada_boost]
    params_list = [rand_forest_params, grad_boost_params, ada_boost_params]
    print "Grid Searching"
    for i, estimator in enumerate(estimators_list):
        print 'Grid Search Loop'
        best_score = 0
        best_params = None
        best_estimator = None
        clf = GridSearchCV(estimator, params_list[i], cv = 5, scoring = 'accuracy', verbose = 2)
        clf.fit(X_reduced, y)
        if clf.best_score_ > best_score:
            best_score = clf.best_score_
            best_params = clf.best_params_
            best_estimator = estimator

    print best_estimator
    print best_params
    print best_score

    print 'returning ideal estimator information'
    return best_estimator, best_params
