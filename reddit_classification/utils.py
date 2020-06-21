import string
import json
import pickle
from pathlib import Path
from kneed import KneeLocator
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk import download, WordNetLemmatizer

download('punkt')
download('stopwords')
download('wordnet')


def tokenize(text, language='english'):
    tokens = word_tokenize(text)
    tokens = [w.lower() for w in tokens]
    table = str.maketrans('', '', string.punctuation)
    words = [w.translate(table) for w in tokens]
    stop_words = set(stopwords.words(language))

    lem = WordNetLemmatizer()
    words = [lem.lemmatize(w, pos="v") for w in words if w not in stop_words and not w.isdigit() and len(w) > 2]

    return ' '.join(words)


def convert_input(x_train, x_test, limit=200, min_df=10):
    tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=min_df)

    x_train_v = tfidf.fit_transform(x_train)
    terms = tfidf.get_feature_names()

    pruned_terms_index, _, _ = prune_vocabulary_until_normalized(x_train_v, terms, limit=limit)
    pruned_terms = [terms[term_index] for term_index in pruned_terms_index]

    tf = CountVectorizer(vocabulary=pruned_terms)

    x_train = tf.fit_transform(x_train).todense()
    x_test = tf.transform(x_test).todense()

    return x_train, x_test, tf


def shuffle_split_data(x, y, test_size=0.25, shuffle=True):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=42, shuffle=shuffle)
    return x_train, x_test, y_train, y_test


def sort_data_by_tfidf_frequency(entries, terms):
    sums = entries.sum(axis=0)

    data = []
    for col in range(len(terms)):
        data.append((col, sums[0, col]))

    ranking = pd.DataFrame(data, columns=['index_term', 'rank'])

    return ranking.sort_values('rank', ascending=False)


def prune_vocabulary_until_normalized(entries, terms, limit=0):
    sorted_data = sort_data_by_tfidf_frequency(entries, terms)
    x, y = sorted_data['index_term'], sorted_data['rank']

    knees_point = []

    while True:
        knee_point = knee_finder(range(len(x)), y)

        if not knee_point or knee_point < limit:
            break

        knees_point.append(knee_point)
        x, y = x[:knee_point], y[:knee_point]

    return x, y, knees_point


def prune_vocabulary(entries, terms):
    sorted_data = sort_data_by_tfidf_frequency(entries, terms)

    x, y = sorted_data['index_term'], sorted_data['rank']

    knee_point = knee_finder(range(len(x)), y)

    return x[:knee_point], y[:knee_point]


def knee_finder(x, y):
    kn = KneeLocator(x, y, curve='convex', direction='decreasing', S=5)
    return kn.knee


def occurrences_per_label(y):
    labels = set(y)
    data = {}
    for label in labels:
        data[label] = y.count(label)

    return data


def read_model(identifier, path='models'):
    full_path = f'{path}/{identifier}'

    with open(f'{full_path}/config.json', 'r') as f:
        config = json.load(f)

    with open(f'{full_path}/model.bin', 'rb') as f:
        model = pickle.load(f)

    with open(f'{full_path}/tokenizer.bin', 'rb') as f:
        tokenizer = pickle.load(f)

    return config, model, tokenizer


def save_model(identifier, config, model, tokenizer, path='models'):
    full_path = f'{path}/{identifier}'
    Path(full_path).mkdir(parents=True, exist_ok=True)

    with open(f'{full_path}config.json', 'w') as f:
        json.dump(config, f)

    with open(f'{full_path}model.bin', 'wb') as f:
        pickle.dump(model.model, f)

    with open(f'{full_path}tokenizer.bin', 'wb') as f:
        pickle.dump(tokenizer, f)


def save_logs(identifier, data, path='logs'):
    Path(path).mkdir(parents=True, exist_ok=True)
    logs_file = f'{path}/{identifier}.logs'

    with open(logs_file, 'w') as f:
        info = {
            'model_identifier': identifier,
            'full_dataset_size': data.get('data_size', -1),
            'Best accuracy': data.get('train_acc', -1),
            'Test accuracy': data.get('test_acc', -1)
        }
        json.dump(info, f)
