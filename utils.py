import json
import pickle
from pathlib import Path
from kneed import KneeLocator
import pandas as pd


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
    kn = KneeLocator(x, y, curve='convex', direction='decreasing', S=10)
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

    with open(f'{full_path}/config.json', 'w') as f:
        json.dump(config, f)

    with open(f'{full_path}/model.bin', 'wb') as f:
        pickle.dump(model.model, f, protocol=pickle.HIGHEST_PROTOCOL)

    with open(f'{full_path}/tokenizer.bin', 'wb') as f:
        pickle.dump(tokenizer, f, protocol=pickle.HIGHEST_PROTOCOL)
