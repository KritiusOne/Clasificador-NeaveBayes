from dataTransform import calculate_word_probabilities
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd


def predict_user(tweet, word_probs_per_user, prior_probs):
    posterior_probs = {user: np.log(prior_probs[user]) for user in prior_probs}
    vocab = word_probs_per_user[list(word_probs_per_user.keys())[0]].keys()

    for word in tweet.split():
        if word in vocab:
            for user in posterior_probs:
                posterior_probs[user] += np.log(
                    word_probs_per_user[user].get(word, 1e-10))

    predicted_user = max(posterior_probs, key=posterior_probs.get)
    return predicted_user


def train_and_evaluate(prior_probs, train_data, test_data, alpha):
    word_probs_per_user = calculate_word_probabilities(train_data)
    summary_list = []
    for _, tweet in test_data.iterrows():
        true_user = tweet['screen_name']
        predicted_user = predict_user(
            tweet['text'], word_probs_per_user, prior_probs)
        summary_list.append(
            {'Usuario_Real': true_user, 'Usuario_Predicho': predicted_user})

    summary_table = pd.DataFrame(summary_list, columns=[
                                 'Usuario_Real', 'Usuario_Predicho'])
    accuracy = accuracy_score(
        summary_table['Usuario_Real'], summary_table['Usuario_Predicho'])
    return accuracy, summary_table
