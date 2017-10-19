import pandas as pd
from textblob.classifiers import NaiveBayesClassifier, MaxEntClassifier



nbc = NaiveBayesClassifier(nb_training)
mec = MaxEntClassifier(me_training)
