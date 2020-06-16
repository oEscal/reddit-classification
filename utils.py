import json
import pickle
from pathlib import Path


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


