import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pickle
from Confluence_page_classification.util import global_config, Util, logger


class DataAnalysis:
    def __init__(self):
        self.data = None

    def load_data(self):
        file_name = (Path.cwd().joinpath(global_config.get_config('path_for_page_storage_data')).
                     joinpath("preprocessed_data.pkl"))
        with open(file_name, 'rb') as file:
            self.data = pickle.load(file)
        logger.info(f"Preprocessed data loaded from {file_name}")

    def check_data_quality(self):

        self.data['label'] = self.data['quality_points'].apply(
            lambda x: 'golden_manuell' if x > 900 else 'archiv' if x <= -900 else '')
        label_mapping = {'golden_manuell': 1, 'archiv': 0, '': -1}
        self.data['label'] = self.data['label'].map(label_mapping)

        # Verteilung der Zielvariablen überprüfen
        label_counts = self.data['label'].value_counts()
        print("Label Counts\n", label_counts)
        plt.figure(figsize=(8, 6))
        label_counts.plot(kind='bar')
        plt.title('Verteilung der Zielvariablen')
        plt.xlabel('Label')
        plt.ylabel('Anzahl')
        plt.show()

        # Verteilung der numerischen Features überprüfen
        numerical_features = self.data.select_dtypes(include=[np.number])
        self.plot_numerical_features(numerical_features)

        # # Verteilung der kategorischen Features überprüfen
        # categorical_features = self.data.select_dtypes(include=[object])
        # for column in categorical_features:
        #     plt.figure(figsize=(8, 6))
        #     self.data[column].value_counts().plot(kind='bar')
        #     plt.title(f'Verteilung von {column}')
        #     plt.xlabel(column)
        #     plt.ylabel('Anzahl')
        #     plt.show()

        # Korrelation der numerischen Features überprüfen
        correlation_matrix = numerical_features.corr()
        print(correlation_matrix)
        plt.figure(figsize=(12, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
        plt.title('Korrelationsmatrix der numerischen Features')
        plt.show()
        self.analyze_correlations(correlation_matrix)

    @staticmethod
    def analyze_correlations(correlation_matrix):
        # Hohe Korrelationen identifizieren
        high_correlation_pairs = []
        threshold = 0.9
        for i in range(len(correlation_matrix.columns)):
            for j in range(i):
                if abs(correlation_matrix.iloc[i, j]) > threshold:
                    colname = correlation_matrix.columns[i]
                    rowname = correlation_matrix.index[j]
                    high_correlation_pairs.append((rowname, colname, correlation_matrix.iloc[i, j]))

        if high_correlation_pairs:
            print("High Correlation Pairs (|correlation| > 0.9):")
            for pair in high_correlation_pairs:
                print(f"{pair[0]} and {pair[1]}: {pair[2]}")
        else:
            print("No high correlations found (|correlation| > 0.9).")

        low_correration_pairs = []
        threshold = 0.1
        for i in range(len(correlation_matrix.columns)):
            for j in range(i):
                if abs(correlation_matrix.iloc[i, j]) < threshold:
                    colname = correlation_matrix.columns[i]
                    rowname = correlation_matrix.index[j]
                    low_correration_pairs.append((rowname, colname, correlation_matrix.iloc[i, j]))
        if low_correration_pairs:
            print("Low Correlation Pairs (|correlation| < 0.1):")
            for pair in low_correration_pairs:
                print(f"{pair[0]} and {pair[1]}: {pair[2]}")
        else:
            print("No low correlations found (|correlation| < 0.1).")

    @staticmethod
    def plot_numerical_features(numerical_features):
        num_features = numerical_features.columns
        num_plots = len(num_features)
        plots_per_figure = 4

        for i in range(0, num_plots, plots_per_figure):
            plt.figure(figsize=(15, 10))
            for j in range(plots_per_figure):
                if i + j < num_plots:
                    plt.subplot(2, 2, j + 1)
                    numerical_features[num_features[i + j]].hist(bins=30)
                    plt.title(num_features[i + j])
            plt.tight_layout()
            plt.show()

    def run(self):
        self.load_data()
        self.check_data_quality()


if __name__ == '__main__':
    data_analysis = DataAnalysis()
    data_analysis.run()
