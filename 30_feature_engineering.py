import pandas as pd
from pathlib import Path
from transformers import pipeline, BertTokenizer, BertModel, BertForTokenClassification
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import pickle
from Confluence_page_classification.util import global_config, logger
from Confluence_page_classification.UtilML import UtilML


class FeatureEngineering:
    def __init__(self):
        self.data = None
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-german-cased')
        self.model = BertModel.from_pretrained('bert-base-german-cased')
        self.lda = None

    def load_data(self):
        file_name = Path.cwd().joinpath(global_config.get_config(
            'path_for_page_storage_data')).joinpath("preprocessed_data.pkl")
        with open(file_name, 'rb') as file:
            self.data = pickle.load(file)
        logger.info(f"Preprocessed data loaded from {file_name}")

    def sentiment_analysis(self):
        sentiment_pipeline = pipeline("sentiment-analysis", model="oliverguhr/german-sentiment-bert")
        self.data['sentiment'] = self.data['text'].apply(lambda x: sentiment_pipeline(x)[0]['label'])
        logger.info("Sentiment analysis completed.")

    def lda_topic_modeling(self):
        vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words=UtilML.get_stopwords_de())
        tfidf_matrix = vectorizer.fit_transform(self.data['text'])
        self.lda = LatentDirichletAllocation(n_components=10, random_state=42)
        lda_topics = self.lda.fit_transform(tfidf_matrix)
        self.data = pd.concat([self.data, pd.DataFrame(lda_topics)], axis=1)
        logger.info("LDA completed.")

    def ner(self):
        ner_pipeline = pipeline("ner", model=BertForTokenClassification.from_pretrained(
            "dbmdz/bert-large-cased-finetuned-conll03-english"),
                                tokenizer="dbmdz/bert-large-cased-finetuned-conll03-english", grouped_entities=True)
        self.data['entities'] = self.data['text'].apply(lambda x: ner_pipeline(x))
        logger.info("NER completed.")

    def readability_analysis(self):
        self.data['readability'] = self.data['text'].apply(UtilML.flesch_kincaid)
        logger.info("Readability analysis completed.")

    def keyphrase_extraction(self):
        self.data['keyphrases'] = self.data['text'].apply(UtilML.extract_keyphrases)
        logger.info("Keyphrase extraction completed.")

    def bert_embeddings(self):
        self.data['bert_embeddings'] = self.data['text'].apply(
            lambda x: UtilML.get_bert_embeddings(x, self.tokenizer, self.model).flatten())
        logger.info("BERT embeddings extraction completed.")

    def save_features(self):
        file_name = Path.cwd().joinpath(global_config.get_config(
            'path_for_page_storage_data')).joinpath("feature_engineered_data.pkl")
        with open(file_name, 'wb') as file:
            pickle.dump(self.data, file)
        logger.info(f"Feature engineered data saved to {file_name}")

    def run(self):
        self.load_data()
        self.sentiment_analysis()
        self.lda_topic_modeling()
        self.ner()
        self.readability_analysis()
        self.keyphrase_extraction()
        self.bert_embeddings()
        self.save_features()


if __name__ == '__main__':
    feature_engineering = FeatureEngineering()
    feature_engineering.run()
