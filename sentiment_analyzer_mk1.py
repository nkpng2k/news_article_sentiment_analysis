from preprocessor_mk1 import TextPreprocessor
from textblob import TextBlob
from textblob.sentiments import NaiveBayesAnalyzer, BaseSentimentAnalyzer
from sklearn.decomposition import LatentDirichletAllocation

class TextSentimentAnalysis(object):

    def __init__(self):
        pass

    def _return_top_words(self, model, feature_names, n_top_words = 50):
        topic_dict = {}
        for topic_idx, topic in enumerate(model.components_):
            topic_top_n_words = [feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]]
            topic_dict[topic_idx] = topic_top_n_words

        return topic_dict

    # --------- All private methods above this line -------

    def lda_dim_reduction(self, vectorized_tokens, vectorizer):
        lda = LatentDirichletAllocation(n_components = 3, learning_method = 'batch').fit(vectorized_tokens)
        feature_names = vectorizer.get_feature_names()
        topic_dict = self._return_top_words(lda, feature_names)

        return topic_dict

    def print_top_words(self, model, feature_names, n_top_words):
        for topic_idx, topic in enumerate(model.components_):
            message = "Topic #%d: " % topic_idx
            message += " ".join([feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]])
            print(message)


if __name__ == '__main__':
    prep = TextPreprocessor()
    article_text = prep.new_article('https://www.washingtonpost.com/local/virginia-politics/reeks-of-subtle-racism-tensions-after-black-candidate-left-off-fliers-in-virginia/2017/10/18/de74c47a-b425-11e7-a908-a3470754bbb9_story.html?utm_term=.2e8be491c0a3')
    vectorizer, vectorized_tokens = prep.generate_vectors(article_text)

    sentiment_analyzer = TextSentimentAnalysis()
    topics_dict = sentiment_analyzer.lda_dim_reduction(vectorized_tokens, vectorizer)

    topics_dict
