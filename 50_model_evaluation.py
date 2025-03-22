from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
import pickle
from Confluence_page_classification.util import global_config, logger


class ModelTraining:
    def __init__(self):
        self.data = None
        self.features = None
        self.labels = None
        self.scaler = None
        self.best_model = None

    def load_data(self):
        file_name = Path.cwd().joinpath(global_config.get_config('path_for_page_storage_data')).joinpath(
            "feature_engineered_data.pkl")
        with open(file_name, 'rb') as file:
            self.data = pickle.load(file)
        logger.info(f"Feature engineered data loaded from {file_name}")

    def prepare_features_labels(self):
        self.data['label'] = self.data['quality_points'].apply(
            lambda x: 'golden_manuell' if x > 900 else 'archiv' if x <= -900 else '')
        label_mapping = {'golden_manuell': 1, 'archiv': 0, '': -1}
        self.data['label'] = self.data['label'].map(label_mapping)

        self.features = self.data.drop(columns=['label'])
        self.labels = self.data['label']

        logger.info("Features and labels prepared.")

    def train_model(self):
        classified_indices = self.labels[self.labels != -1].index
        training_features = self.features.loc[classified_indices]
        training_labels = self.labels.loc[classified_indices]

        x_train, x_test, y_train, y_test = train_test_split(training_features, training_labels, test_size=0.2,
                                                            random_state=42)
        self.scaler = StandardScaler()
        x_train_scaled = self.scaler.fit_transform(x_train)
        x_test_scaled = self.scaler.transform(x_test)

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

        conf_matrix = confusion_matrix(y_test, y_pred)
        logger.info(f"Confusion Matrix:\n{conf_matrix}")

        self.save_model()

    def save_model(self):
        file_name = Path.cwd().joinpath(global_config.get_config('path_for_page_storage_data')).joinpath(
            "trained_model.pkl")
        with open(file_name, 'wb') as file:
            pickle.dump(self.best_model, file)
        logger.info(f"Trained model saved to {file_name}")

    def run(self):
        self.load_data()
        self.prepare_features_labels()
        self.train_model()


if __name__ == '__main__':
    model_training = ModelTraining()
    model_training.run()
