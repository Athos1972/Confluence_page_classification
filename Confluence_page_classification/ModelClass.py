import pandas as pd
from pathlib import Path
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from transformers import pipeline, BertTokenizer, BertModel, BertForTokenClassification
import shap
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from Confluence_page_classification.util import global_config, Util, logger
from Confluence_page_classification.UtilFileHandling import UtilFilehandling
from Confluence_page_classification.UtilML import UtilML
import pickle


class PageClassification:
    def __init__(self):
        self.data = None
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-german-cased')
        self.model = BertModel.from_pretrained('bert-base-german-cased')
        self.lda = None
        self.scaler = None
        self.best_model = None
        self.prev_data = None
        self.step = 1
        self.features = None
        self.labels = None

    def load_data(self):
        path_to_pickle = Path.cwd().joinpath(global_config.get_config('path_for_page_storage_data'))
        latest_pickle = UtilFilehandling.get_latest_file_in_folder(str(path_to_pickle), "pkl")
        with open(latest_pickle, 'rb') as file:
            data_pickled = pickle.load(file)
            self.data = pd.DataFrame([vars(page) for page in data_pickled.values()])
        logger.info(f"Data loaded from pickle file {latest_pickle}.")
        self.export_step()

    def preprocess_text(self):
        self.data.rename(columns={"plain_text_content": "text"}, inplace=True)
        self.data.drop(columns=['content'], inplace=True)
        self.data['text'] = self.data['text'].apply(lambda x: UtilML.truncate_text(x, self.tokenizer))
        logger.info("Text truncation completed.")
        self.export_step()

    def sentiment_analysis(self):
        sentiment_pipeline = pipeline("sentiment-analysis", model="oliverguhr/german-sentiment-bert")
        self.data['sentiment'] = self.data['text'].apply(lambda x: sentiment_pipeline(x)[0]['label'])
        logger.info("Sentiment analysis completed.")
        self.export_step()

    def check_data_quality(self):
        # Fehlende Werte überprüfen
        missing_values = self.data.isnull().sum()
        print(missing_values)

        # Verteilung der Zielvariablen überprüfen
        label_counts = self.data['labels'].value_counts()
        print(label_counts)
        plt.figure(figsize=(8, 6))
        label_counts.plot(kind='bar')
        plt.title('Verteilung der Zielvariablen')
        plt.xlabel('Label')
        plt.ylabel('Anzahl')
        plt.show()

        # Verteilung der numerischen Features überprüfen
        numerical_features = self.data.select_dtypes(include=[np.number])
        numerical_features.hist(bins=30, figsize=(15, 10))
        plt.show()

        # Verteilung der kategorischen Features überprüfen
        categorical_features = self.data.select_dtypes(include=[object])
        for column in categorical_features:
            plt.figure(figsize=(8, 6))
            self.data[column].value_counts().plot(kind='bar')
            plt.title(f'Verteilung von {column}')
            plt.xlabel(column)
            plt.ylabel('Anzahl')
            plt.show()

        # Korrelation der numerischen Features überprüfen
        correlation_matrix = numerical_features.corr()
        print(correlation_matrix)
        plt.figure(figsize=(12, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
        plt.title('Korrelationsmatrix der numerischen Features')
        plt.show()

    def lda_topic_modeling(self):
        vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words=UtilML.get_stopwords_de())
        tfidf_matrix = vectorizer.fit_transform(self.data['text'])
        self.lda = LatentDirichletAllocation(n_components=10, random_state=42)
        lda_topics = self.lda.fit_transform(tfidf_matrix)
        # Rename the columns in LDA_Topcis to "LDA" + column name
        lda_topics = pd.DataFrame(lda_topics, columns=[f"LDA_{i}" for i in range(lda_topics.shape[1])])
        self.data = pd.concat([self.data, pd.DataFrame(lda_topics)], axis=1)
        logger.info("LDA completed.")
        self.export_step()

    def ner(self):
        ner_pipeline = pipeline("ner", model=BertForTokenClassification.from_pretrained(
            "dbmdz/bert-large-cased-finetuned-conll03-english"),
                                tokenizer="dbmdz/bert-large-cased-finetuned-conll03-english", grouped_entities=True)
        self.data['entities'] = self.data['text'].apply(lambda x: ner_pipeline(x))
        logger.info("NER completed.")
        self.export_step()

    def readability_analysis(self):
        self.data['readability'] = self.data['text'].apply(UtilML.flesch_kincaid)
        logger.info("Readability analysis completed.")
        self.export_step()

    def keyphrase_extraction(self):
        self.data['keyphrases'] = self.data['text'].apply(UtilML.extract_keyphrases)
        logger.info("Keyphrase extraction completed.")
        self.export_step()

    def bert_embeddings(self):
        self.data['bert_embeddings'] = self.data['text'].apply(
            lambda x: UtilML.get_bert_embeddings(x, self.tokenizer, self.model).flatten())
        logger.info("BERT embeddings extraction completed.")
        self.export_step()

    def prepare_features_labels(self):
        self.data['label'] = self.data['quality_points'].apply(
            lambda x: 'golden_manuell' if x > 900 else 'archiv' if x <= -900 else '')
        label_mapping = {'golden_manuell': 1, 'archiv': 0, '': -1}
        self.data['label'] = self.data['label'].map(label_mapping)

        self.features = pd.concat([self.data[['h1_count', 'h2_count', 'h3_count', 'tasks_open', 'tasks_closed',
                                              'page_properties', 'page_properties_report', 'user_mentions',
                                              'dates_in_text', 'incoming_links_count', 'pure_length', 'age_in_days',
                                              'last_updated_in_days', 'version_count', 'ancestors_count',
                                              'children_count', 'table_count', 'rows_count',
                                              'jira_links_count', 'confluence_links_count', 'image_in_page_count',
                                              'quality_points', 'plantuml_macros_count']],
                                   # the LDA-Fieldnames:
                                   self.data[[col for col in self.data.columns if 'LDA' in col]],
                                   pd.get_dummies(self.data['sentiment']),
                                   self.data[['readability']],
                                   # The columns from bert_embeddings should be named "bert_0", "bert_1", ...
                                   pd.DataFrame(self.data['bert_embeddings'].tolist(),
                                                columns=[f"bert_{i}" for i in range(768)])],
                                  axis=1)

        self.labels = self.data['label']

        logger.info("Features and labels prepared.")
        self.export_step()

    def train_model(self):
        classified_indices = self.labels[self.labels != -1].index
        training_features = self.features.loc[classified_indices]
        training_labels = self.labels.loc[classified_indices]

        x_train, x_test, y_train, y_test = train_test_split(training_features, training_labels, test_size=0.2,
                                                            random_state=42)
        self.scaler = StandardScaler()
        x_train_scaled = self.scaler.fit_transform(x_train)
        x_test_scaled = self.scaler.transform(x_test)

        # Hyperparameter tuning - this combination lead to overfitting of the model. Therefore, the grid was reduced.
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }

        param_grid = {
            'n_estimators': [10, 50, 100],
            'max_depth': [5, 10, 15],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }

        grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5)
        grid_search.fit(x_train_scaled, y_train)
        self.best_model = grid_search.best_estimator_

        cv_scores = cross_val_score(self.best_model, x_train_scaled, y_train, cv=5)
        logger.info(f"Cross-validation scores: {cv_scores}")

        y_pred = self.best_model.predict(x_test_scaled)
        logger.info("\n" + classification_report(y_test, y_pred))
        self.export_step()

        # Confusion matrix and misclassified samples analysis
        conf_matrix = confusion_matrix(y_test, y_pred)
        logger.info(f"Confusion Matrix:\n{conf_matrix}")

        misclassified_indices = [i for i, (true, pred) in enumerate(zip(y_test, y_pred)) if true != pred]
        misclassified_samples = x_test[misclassified_indices]
        logger.info(f"Misclassified samples:\n{misclassified_samples}")

        # SHAP analysis
        explainer = shap.TreeExplainer(self.best_model)
        shap_values = explainer.shap_values(x_test_scaled)
        shap.summary_plot(shap_values, x_test, plot_type="bar")
        plt.savefig(Path(global_config.get_config('path_for_page_storage_data')).joinpath(
            f"shap_summary_plot_{Util.get_current_date_formatted_for_filename()}.png"))

    def classify_remaining(self):
        remaining_indices = self.labels[self.labels == -1].index
        remaining_features = self.features.loc[remaining_indices]

        if not remaining_features.empty:
            remaining_features_scaled = self.scaler.transform(remaining_features)
            self.data.loc[remaining_indices, 'predicted_label'] = self.best_model.predict(remaining_features_scaled)
        else:
            self.data['predicted_label'] = []

        reverse_label_mapping = {1: 'golden_ml', 0: 'archiv_ml', -1: ''}
        self.data['predicted_label'] = self.data['predicted_label'].map(reverse_label_mapping)

        file_name = Path(global_config.get_config('path_for_page_storage_data')).joinpath(
            f"ml_final_results_mapped_{Util.get_current_date_formatted_for_filename()}.xlsx")
        Util.write_dataframe_to_excel(dfs=self.data, filename=str(file_name), sheetname="ML-Results-Mapped")
        logger.info(f"Final results with mapped labels saved in {file_name}")

    def export_step(self):
        if self.prev_data is not None:
            Util.write_delta_dataframe_to_excel(prev_df=self.prev_data,
                                                current_df=self.data,
                                                step_number=self.step)
        self.prev_data = self.data.copy()
        self.step += 1

    def run(self):
        self.load_data()
        self.preprocess_text()
        self.check_data_quality()
        self.sentiment_analysis()
        self.lda_topic_modeling()
        self.ner()
        self.readability_analysis()
        self.keyphrase_extraction()
        self.bert_embeddings()
        self.prepare_features_labels()
        self.train_model()
        self.classify_remaining()
