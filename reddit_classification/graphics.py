import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

from reddit_classification.utils import occurrences_per_label, knee_finder, sort_data_by_tfidf_frequency, \
    prune_vocabulary_until_normalized
from reddit_classification.data_treatment import read_data
from reddit_classification.classifier import shuffle_split_data, convert_input, tokenize


def data_distribution(title, labels, counters, path):
    y_pos = np.arange(len(labels))
    plt.bar(y_pos, counters, alpha=0.5)
    plt.xticks(y_pos, labels, rotation=90)
    plt.ylabel('Occurrences')
    plt.title(title)

    plt.savefig(path, bbox_inches='tight', dpi=400)
    plt.clf()


def vocabulary_plot(path, title, x, terms, show_knee=True):
    sorted_data = sort_data_by_tfidf_frequency(x, terms)

    x, y = range(len(sorted_data['index_term'])), sorted_data['rank']

    knee_point = knee_finder(x, y)

    plt.ylabel('TFIDF Frequency')
    plt.title(title)
    plt.plot(x, y)
    if show_knee:
        plt.vlines(knee_point, plt.ylim()[0], plt.ylim()[1], linestyles='dashed')
    plt.savefig(path, bbox_inches='tight', dpi=400)
    plt.clf()


def pruned_vocabulary_frequencies(path, title, x, terms, show_knee=True, limit=0):
    sorted_data = sort_data_by_tfidf_frequency(x, terms)

    _, _, knees_points = prune_vocabulary_until_normalized(x, terms, limit)

    data_size = [len(sorted_data['rank'])] + knees_points
    for i in range(len(knees_points)):
        knee_point = knees_points[i]
        x, y = range(data_size[i]), sorted_data['rank'][:data_size[i]]

        plt.ylabel('TFIDF Frequency')
        plt.title(f'{title} -> Data size : {data_size[i]}')
        plt.plot(x, y)
        if show_knee:
            plt.vlines(knee_point, plt.ylim()[0], plt.ylim()[1], linestyles='dashed')
            plt.text(knee_point, max(y), str(knee_point))
        file_path = f'{path}/data_size_{knee_point}.png'
        plt.savefig(file_path, bbox_inches='tight', dpi=400)
        plt.clf()


def main():
    x, y = read_data('data/pruned_5_entries.tsv')

    occurrences_per_label_dict = occurrences_per_label(y)

    data_distribution("Full data distribution", list(occurrences_per_label_dict.keys()),
                      list(occurrences_per_label_dict.values()), "graphics/full_data_distribution.png")

    x_train, x_test, y_train, y_test = shuffle_split_data(x, y, 0.2, True)

    train_occurrences_per_label_dict = occurrences_per_label(y_train)
    test_occurrences_per_label_dict = occurrences_per_label(y_test)

    data_distribution("Train data distribution", list(train_occurrences_per_label_dict.keys()),
                      list(train_occurrences_per_label_dict.values()), "graphics/train_data_distribution.png")

    data_distribution("Test data distribution", list(test_occurrences_per_label_dict.keys()),
                      list(test_occurrences_per_label_dict.values()), "graphics/test_data_distribution.png")

    x_train = [tokenize(text) for text in x_train]
    x_test = [tokenize(text) for text in x_test]

    v_x_train, v_x_test, vocab_size, vectorizer = convert_input(x_train, x_test)

    vocabulary_plot('graphics/train_pruned_vocabulary_tfidf_frequencies.png', "Train set tfidf frequencies",
                    v_x_train, vectorizer.get_feature_names(), show_knee=False)

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
    x_train = vectorizer.fit_transform(x_train).todense()
    x_test = vectorizer.transform(x_test).todense()

    vocabulary_plot('graphics/train_vocabulary_tfidf_frequencies.png', "Train set tfidf frequencies",
                    x_train, vectorizer.get_feature_names())
    vocabulary_plot('graphics/test_vocabulary_tfidf_frequencies.png', "Test set tfidf frequencies",
                    x_test, vectorizer.get_feature_names())

    pruned_vocabulary_frequencies('graphics/vocabulary_pruned_until_normalize', "Train set tfidf frequencies", x_train,
                                  vectorizer.get_feature_names())


if __name__ == '__main__':
    main()
