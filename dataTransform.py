from Types import FILE_NAME, columns
import pandas as pd


def getDataSet():
    df = pd.read_csv(FILE_NAME, names=columns, delimiter=',')

    df['text'] = df['text'].str.replace('[^a-zA-Z\s]', '', regex=True)
    return df


def calculate_word_probabilities(train_data):
    word_counts_per_user = {}
    vocab = set()

    for _, tweet in train_data.iterrows():
        user = tweet['screen_name']
        text = tweet['text']

        if user not in word_counts_per_user:
            word_counts_per_user[user] = {}

        for word in text.split():
            word_counts_per_user[user][word] = word_counts_per_user[user].get(
                word, 0) + 1
            vocab.add(word)

    word_probs_per_user = {}
    vocab_size = len(vocab)

    for user, word_counts in word_counts_per_user.items():
        word_probs_per_user[user] = {}
        total_words = sum(word_counts.values()) + vocab_size

        for word in vocab:
            word_count = word_counts.get(word, 0) + 1
            word_probs_per_user[user][word] = word_count / total_words

    return word_probs_per_user
