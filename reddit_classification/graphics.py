import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

from .utils import occurrences_per_label, knee_finder, sort_data_by_tfidf_frequency, \
    prune_vocabulary_until_normalized, shuffle_split_data, convert_input, tokenize
from .data_treatment import read_data
from .classifier import pick_best_model


def data_distribution(title, labels, counters, save=True, path=None):
    y_pos = np.arange(len(labels))
    plt.bar(y_pos, counters, alpha=0.5)
    plt.xticks(y_pos, labels, rotation=90)
    plt.xlabel("Label")
    plt.ylabel("Number of Examples")
    plt.title(title)

    if save:
        plt.savefig(path, bbox_inches='tight', dpi=400)
        plt.clf()


def all_data_distribution(y, path, file_name, with_cv_data=False):
    def create_graph(title, labels, counters, ax):
        y_pos = np.arange(len(labels))

        ax.bar(y_pos, counters, alpha=0.5)
        ax.set_title(title)
        ax.set_xlabel("Label")
        ax.set_ylabel("Number of Examples")
        ax.set_xticks(y_pos)
        ax.set_xticklabels(labels, rotation='vertical')

    if with_cv_data:
        fig, axs = plt.subplots(1, 3, figsize=(23, 6))

        train_split_index = int(len(y) * 0.6)
        cv_split_index = train_split_index + int(len(y) * 0.2)
        y_train, y_cv, y_test = y[:train_split_index], y[train_split_index:cv_split_index], y[cv_split_index:]
    else:
        fig, axs = plt.subplots(1, 2, figsize=(15, 7))

        train_split_index = int(len(y) * 0.8)
        y_train, y_test = y[:train_split_index], y[train_split_index:]

    occurrences_per_label_dict = occurrences_per_label(y_train)
    create_graph("Train/Cross validation data distribution" if not with_cv_data else "Train data distribution",
                 list(occurrences_per_label_dict.keys()), list(occurrences_per_label_dict.values()), axs[0])

    if with_cv_data:
        occurrences_per_label_dict = occurrences_per_label(y_cv)
        create_graph("Cross validation data distribution", list(occurrences_per_label_dict.keys()),
                     list(occurrences_per_label_dict.values()), axs[1])

    occurrences_per_label_dict = occurrences_per_label(y_test)
    create_graph("Test data distribution", list(occurrences_per_label_dict.keys()),
                 list(occurrences_per_label_dict.values()), axs[1] if not with_cv_data else axs[2])

    plt.tight_layout()

    plt.savefig(f"{path}/{file_name}")


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


def plot_accuracy_function(history, path):
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig(path, bbox_inches='tight', dpi=400)
    plt.clf()


def plot_accuracy_over_var(study_for_data, accuracy, x_label, path, hold_on=False, max_value=True, legend=None):
    plt.plot(study_for_data, accuracy, marker='o')
    plt.grid()
    plt.title("Accuracy")
    plt.xlabel(x_label)
    plt.ylabel("Accuracy")

    if max_value:
        max_accuracy = max(accuracy)
        max_accuracy_index = accuracy.index(max_accuracy)
        max_accuracy_data = study_for_data[max_accuracy_index]
        plt.annotate(f"({max_accuracy_data}, {max_accuracy:.3})",
                     xy=(max_accuracy_data + max(study_for_data) / 50, max_accuracy + min(accuracy) / 50))

    if legend:
        plt.legend(legend)

    if not hold_on:
        plt.savefig(f"{path}/accuracy_{x_label}.png")


def main():
    graphics_path = "graphics"
    output_path = "models"

    number_classes = 50
    if number_classes:
        X, y = read_data(f"data/pruned_{number_classes}_entries.tsv")
    else:
        X, y = read_data()

    all_data_distribution(y, "graphics", "all_distribution.png")
    all_data_distribution(y, "graphics", "all_distribution_cv.png",  with_cv_data=True)

    identifier = "modelo"
    return
    history, _, _, _ = pick_best_model(identifier, X, y, output_path)

    plot_accuracy_function(history, f"{graphics_path}/accuracy_function.png")

    occurrences_per_label_dict = occurrences_per_label(y)

    data_distribution("Full data distribution", list(occurrences_per_label_dict.keys()),
                      list(occurrences_per_label_dict.values()), path=f"{graphics_path}/full_data_distribution.png")

    x_train, x_test, y_train, y_test = shuffle_split_data(X, y, 0.2, True)

    train_occurrences_per_label_dict = occurrences_per_label(y_train)
    test_occurrences_per_label_dict = occurrences_per_label(y_test)

    data_distribution("Train data distribution", list(train_occurrences_per_label_dict.keys()),
                      list(train_occurrences_per_label_dict.values()), path=f"{graphics_path}/train_data_distribution.png")

    data_distribution("Test data distribution", list(test_occurrences_per_label_dict.keys()),
                      list(test_occurrences_per_label_dict.values()), path=f"{graphics_path}/test_data_distribution.png")

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
