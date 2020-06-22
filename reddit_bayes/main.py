import pickle
import time

import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB

from reddit_classification.graphics import data_distribution
from reddit_classification.utils import convert_input, occurrences_per_label
from reddit_classification.data_treatment import read_data


def main():
	"""
	from https://www.kaggle.com/mswarbrickjones/starter-naive-bayes-benchmark-p-at-5-0-88
	"""

	number_classes = 20
	if number_classes:
		X, y = read_data(f"data/pruned_{number_classes}_entries.tsv")
	else:
		X, y = read_data()

	train_split_index = int(len(X) * 0.6)
	cv_split_index = train_split_index + int(len(X) * 0.2)
	# 20% for cross validation and 20% for test

	X_train, X_cv = X[:train_split_index], X[train_split_index:cv_split_index]
	y_train, y_cv, y_test = y[:train_split_index], y[train_split_index:cv_split_index], y[cv_split_index:]

	occurrences_per_label_dict = occurrences_per_label(y_train)
	print(len(occurrences_per_label_dict))
	data_distribution("Train data distribution", list(occurrences_per_label_dict.keys()),
	                  list(occurrences_per_label_dict.values()), f"reddit_bayes/graphs/train_data_distribution.png")
	occurrences_per_label_dict = occurrences_per_label(y_cv)
	print(len(occurrences_per_label_dict))
	data_distribution("Cross Validation data distribution", list(occurrences_per_label_dict.keys()),
	                  list(occurrences_per_label_dict.values()), f"reddit_bayes/graphs/cv_data_distribution.png")
	occurrences_per_label_dict = occurrences_per_label(y_test)
	print(len(occurrences_per_label_dict))
	data_distribution("Test data distribution", list(occurrences_per_label_dict.keys()),
	                  list(occurrences_per_label_dict.values()), f"reddit_bayes/graphs/test_data_distribution.png")

	label_encoder = LabelEncoder()
	label_encoder.fit(y_train)
	y_train = label_encoder.transform(y_train)
	y_cv = label_encoder.transform(y_cv)

	# extract features
	X_train, X_cv, token_frequencies = convert_input(X_train, X_cv, limit=1000, min_df=1)

	# save token frequencies
	with open(f"reddit_bayes/token_frequencies/token_frequencies_{number_classes}", 'wb') as file:
		pickle.dump(token_frequencies, file)

	# with open(f"reddit_bayes/token_frequencies/token_frequencies_{number_classes}", 'rb') as file:
	# 	token_frequencies = pickle.load(file)
	# 	X_train = token_frequencies.fit_transform(X_train).todense()
	# 	X_cv = token_frequencies.transform(X_cv).todense()

	# normalize
	X_train = X_train / np.max(X_train)
	X_cv = X_cv / np.max(X_cv)

	# train
	initial_alpha = 1e-10
	while initial_alpha <= 1:
		for i in [1, 5]:
			alpha = initial_alpha * i

			initial_time = time.time()
			model = MultinomialNB(alpha=alpha)
			model.fit(X_train, y_train)

			# save model
			with open(f"reddit_bayes/models/model_{number_classes}_{format(alpha, '.10g')}", 'wb') as file:
				pickle.dump(model, file)

			# with open(f"reddit_bayes/models/model_{number_classes}_{alpha}", 'rb') as file:
			# 	model = pickle.load(file)

			y_pred_proba = model.predict_proba(X_cv)
			y_pred = np.argmax(y_pred_proba, axis=1)

			print(f"precision for alpha {format(alpha, '.10g')} = {np.mean(y_cv == y_pred)}")

			print(time.time() - initial_time)
		initial_alpha *= 10


if __name__ == "__main__":
	main()
