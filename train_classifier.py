import pandas as pd
from textblob.classifiers import NaiveBayesClassifier, MaxEntClassifier

pol_df = pd.read_csv('data/politics.tsv', sep = '\t', header = None)
adj_df = pd.read_csv('data/2000.tsv', sep = '\t', header = None)
pol_df.columns = ['word', 'labels', 'etc']
adj_df.columns = ['word', 'labels', 'etc']

pol_mask = pol_df['labels'] >= 0.0
adj_mask = adj_df['labels'] >= 0.0

adj_labels = adj_df['labels'].copy()
adj_labels[adj_mask] = 'pos'
adj_labels[~adj_mask] = 'neg'
adj_df['etc'] = adj_labels

pol_labels = pol_df['labels'].copy()
pol_labels[pol_mask] = 'pos'
pol_labels[~pol_mask] = 'neg'
pol_df['etc'] = pol_labels

nb_training = set()

for i, row in pol_df.iterrows():
    nb_training.add((row[0], row[2]))
for i, row in adj_df.iterrows():
    nb_training.add((row[0], row[2]))

nbc = NaiveBayesClassifier(nb_training)
prob_dist = nbc.prob_classify('trump hates racism')
prob_dist.max()
prob_dist.prob('neg')

nb_name = 'naivebayesclassifier.pkl'
with open(nb_name, 'wb') as f:
    pickle.dump(nbc, f)


with open('naivebayesclassifier.pkl') as f:
    pickled_nbc = pickle.load(f)
