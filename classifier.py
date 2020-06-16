import string

from nltk import download, WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras import Sequential, layers
from utils import read_model, save_model
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
import json

download('punkt')
download('stopwords')
download('wordnet')


def tokenize(text, language='english'):
	tokens = word_tokenize(text)
	tokens = [w.lower() for w in tokens]
	table = str.maketrans('', '', string.punctuation)
	words = [w.translate(table) for w in tokens]
	stop_words = set(stopwords.words(language))

	lemmatizer = WordNetLemmatizer()
	words = [lemmatizer.lemmatize(w, pos="v") for w in words if w not in stop_words and len(w) > 2]

	return ' '.join(words)


def convert_input(x_train, x_test, max_len, num_words=5000):
	"""
	More info:
		- https://keras.io/api/preprocessing/text/
		- https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/sequence/pad_sequences
	"""
	tokenizer = Tokenizer(num_words=num_words)
	tokenizer.fit_on_texts(x_train)  # Train tokenizer with train set
	vocab_size = len(tokenizer.word_index) + 1

	x_train = pad_sequences(tokenizer.texts_to_sequences(x_train), maxlen=max_len)
	x_test = pad_sequences(tokenizer.texts_to_sequences(x_test), maxlen=max_len)

	return x_train, x_test, vocab_size, tokenizer


def shuffle_split_data(x, y, test_size=0.25, shuffle=True):
	x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=0, shuffle=shuffle)
	return x_train, x_test, y_train, y_test


def create_model(num_filters, kernel_size, vocab_size, embedding_dim, max_len):
	model = Sequential()
	model.add(layers.Embedding(vocab_size, embedding_dim, input_length=max_len))
	model.add(layers.Conv1D(num_filters, kernel_size, activation='relu'))
	model.add(layers.GlobalMaxPooling1D())
	model.add(layers.Dense(10, activation='relu'))
	model.add(layers.Dense(1, activation='sigmoid'))
	model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
	return model


def pick_best_model(identifier, x, y, max_len=100, param_grid=None, embedding_dim=50, epochs=20, n_jobs=-1,
                    batch_size=10, save_logs=True):
	x = [tokenize(text) for text in x]
	x_train, x_test, y_train, y_test = shuffle_split_data(x, y, 0.2, True)
	x_train, x_test, vocab_size, tokenizer = convert_input(x_train, x_test, max_len)

	if param_grid is None:
		param_grid = {
			"num_filters": [32, 64, 128, 256],
			"kernel_size": [3, 5, 7],
			"vocab_size": [vocab_size],
			"embedding_dim": [embedding_dim],
			"max_len": [max_len]
		}

		model = KerasClassifier(build_fn=create_model, epochs=epochs, batch_size=batch_size, verbose=True)

		grid = RandomizedSearchCV(estimator=model, param_distributions=param_grid, cv=min(4, len(y_train)), verbose=1,
		                          n_iter=5, n_jobs=n_jobs)

		grid_result = grid.fit(x_train, y_train)
		best_model = grid_result.best_estimator_

		config = {
			'epochs': epochs,
			'batch_size': batch_size,
			'max_len': max_len,
			'padding': 'post'
		}

		if save_logs:
			logs_file = f'logs/{identifier}.logs'
			with open(logs_file, 'a') as f:
				info = {
					'model_identifier': identifier,
					'full_dataset_size': len(x),
					'Best accuracy': grid_result.best_score_,
					'Test accuracy': grid.score(x_test, y_test)
				}
				json.dump(info, f)

		save_model(identifier, config, best_model, tokenizer)

		return config, best_model, tokenizer


def predict(identifier, x):
	config, model, tokenizer = read_model(identifier)

	classifier = KerasClassifier(build_fn=create_model, epochs=config['epochs'], batch_size=config['batch_size'],
	                             verbose=True)

	classifier.model = model
	x = [tokenize(text) for text in x]
	x = pad_sequences(tokenizer.texts_to_sequences(x), padding=config['padding'], maxlen=config['max_len'])

	return model.predict(x)
