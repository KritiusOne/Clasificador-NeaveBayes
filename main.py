from sklearn.model_selection import train_test_split
from collections import defaultdict
from dataTransform import getDataSet
from evaluation import train_and_evaluate
import numpy as np
import pandas as pd
from Types import alphas


df = getDataSet()
df_unique = df.groupby("screen_name").agg(
    {"screen_name": "first", "text": "first"})
df_purgate = df[df["screen_name"].duplicated()]


def getClasificacion(df):
    train_data, test_data = train_test_split(
        df, test_size=0.2, random_state=42)

    alpha_accuracies = defaultdict(list)

    for alpha in alphas:
        prior_probs = train_data['screen_name'].value_counts(
            normalize=True).to_dict()
        accuracy, _ = train_and_evaluate(
            prior_probs, train_data, test_data, alpha)
        alpha_accuracies[alpha].append(accuracy)

    mean_accuracies = {alpha: np.mean(acc)
                       for alpha, acc in alpha_accuracies.items()}
    best_alpha = max(mean_accuracies, key=mean_accuracies.get)
    best_accuracy = mean_accuracies[best_alpha]

    prior_probs = train_data['screen_name'].value_counts(
        normalize=True).to_dict()
    accuracy, summary_table = train_and_evaluate(
        prior_probs, train_data, test_data, best_alpha)
    # esto de aca es para que el pandas no resuma la tabla sino que la muestre toda no es tan importante
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)
    accuracy = accuracy*100
    print("\nSummary Table:")
    print(summary_table)

    print(f"\nPrecisión : {accuracy:.8f}%")


getClasificacion(df_unique)
getClasificacion(df_purgate)
