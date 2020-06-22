import pickle
import time

import numpy as np
from sklearn.ensemble import VotingClassifier
import matplotlib as mp
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB


from reddit_classification.utils import convert_input
from reddit_classification.data_treatment import read_data


def evaluate_model(model, X, y, X_test, y_test, target_names=None):
	scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
	scores_test = cross_val_score(model, X_test, y_test, cv=5, scoring='accuracy')

	print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std()))
	print("Accuracy test: %0.2f (+/- %0.2f)" % (scores_test.mean(), scores_test.std()))

	print("Test classification report: ")
	if target_names is None:
		target_names = model.classes_
	# print(classification_report(y_test, model.predict(X_test), target_names=target_names))
	print("Test confusion matrix: ")
	# print_confusion_matrix(confusion_matrix(y_test, model.predict(X_test)), class_names=target_names)


def main():
	"""
	from https://www.kaggle.com/balatmak/text-classification-pipeline-newsgroups20
	"""

	number_classes = 50
	if number_classes:
		X, y = read_data(f"data/pruned_{number_classes}_entries.tsv")
	else:
		X, y = read_data()

	train_split_index = int(len(X) * 0.6)
	cv_split_index = train_split_index + int(len(X) * 0.2)
	# 20% for cross validation and 20% for test

	X_train, X_cv = X[:train_split_index], X[train_split_index:cv_split_index]
	y_train, y_cv = y[:train_split_index], y[train_split_index:cv_split_index]

	label_encoder = LabelEncoder()
	label_encoder.fit(y_train)
	y_train = label_encoder.transform(y_train)
	y_cv = label_encoder.transform(y_cv)

	# extract features
	# X_train, X_test, token_frequencies = convert_input(X_train, X_test, limit=1000, min_df=5)

	# save token frequencies
	# with open(f"reddit_bayes/token_frequencies/token_frequencies_{number_classes}", 'wb') as file:
	# 	pickle.dump(token_frequencies, file)

	with open(f"reddit_bayes/token_frequencies/token_frequencies_{number_classes}", 'rb') as file:
		token_frequencies = pickle.load(file)
		X_train = token_frequencies.fit_transform(X_train).todense()
		X_cv = token_frequencies.transform(X_cv).todense()

	# normalize
	X_train = X_train / np.max(X_train)
	X_cv = X_cv / np.max(X_cv)

	c_s = [1, 10, 100, 1000, 10000]
	for c in c_s:
		initial_time = time.time()
		model = LogisticRegression(C=c, multi_class='ovr', max_iter=1000)

		model.fit(X_train, y_train)

		with open(f"reddit_logistic/models/model_{number_classes}_{c}", 'wb') as file:
			pickle.dump(model, file)

		evaluate_model(model, X_train, y_train, X_cv, y_cv)

		print(f"\ntime: {time.time() - initial_time}\n\n\n")


if __name__ == "__main__":
	main()
