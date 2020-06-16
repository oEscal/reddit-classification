import csv
from sklearn.model_selection import train_test_split
import json
import pickle
from pathlib import Path


def read_data(file_path='data/entries.tsv'):
	x, y = [], []

	with open(file_path) as f:
		rows = list(csv.reader(f, delimiter='\t', quotechar='"'))
		for row in rows:
			identifier, theme, title, text = row
			x.append(text)
			y.append(theme)

	return x, y


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


def shuffle_split_data(x, y, test_size=0.25, shuffle=True):
	x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=0, shuffle=shuffle)
	return x_train, x_test, y_train, y_test
