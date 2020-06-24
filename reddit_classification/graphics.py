import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from .utils import knee_finder, sort_data_by_tfidf_frequency, \
    prune_vocabulary_until_normalized, occurrences_per_label
from .data_treatment import read_data
import math


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

    all_data = []

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
        all_data.append((x, y, knee_point, data_size[i]))

    n_cols = 2
    fig, ax = plt.subplots(nrows=math.ceil(len(all_data) / n_cols), ncols=n_cols, figsize=(15, 15))
    index = 0
    for row in ax:
        for col in row:
            if len(all_data) > index:
                x, y, knee_point, data_size = all_data[index]
                col.plot(x, y)
                col.set_title(f'{title} -> Data size : {data_size}')
                col.set(ylabel='TFIDF Frequency')
                if show_knee:
                    col.vlines(knee_point, min(y), max(y), linestyles='dashed')
                    col.text(knee_point, max(y), str(knee_point))
            index += 1

    file_path = f'{path}/all_graphics.png'
    plt.savefig(file_path, bbox_inches='tight', dpi=400)
    plt.clf()


def plot_accuracy_function(history, file_path, title, path='graphics/accuracy_function', l='val'):
    Path(path).mkdir(parents=True, exist_ok=True)
    plt.plot(history['accuracy'])
    plt.plot(history['val_accuracy'])
    plt.title(title)
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', l], loc='upper left')
    plt.savefig(f'{path}/{file_path}', bbox_inches='tight', dpi=400)
    plt.clf()


def merge_accuracy_fold(all_data, path='graphics/accuracy_function'):
    Path(path).mkdir(parents=True, exist_ok=True)

    n_cols = 2
    fig, ax = plt.subplots(nrows=math.ceil(len(all_data) / n_cols), ncols=n_cols, figsize=(20, 20))
    index = 0
    for row in ax:
        for col in row:
            if len(all_data) > index:
                x, y = all_data[index]
                col.plot(x)
                col.plot(y)
                col.set_title(f'Model accuracy fold {index + 1}')
                col.set(xlabel='Epoch', ylabel='Accuracy')
                col.legend(['train', 'val'], loc='upper left')
            index += 1

    file_path = f'{path}/merged_fold_graphics.png'

    plt.savefig(file_path, bbox_inches='tight', dpi=400)
    plt.clf()


def sentiment_vs_no_sentiment_accuracy(without_sentiment, with_sentiment, path='graphics/'):
    labels = ['Train set accuracy', 'Test set accuracy']
    x = np.arange(len(labels))  # the label locations
    width = 0.1  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width / 2, without_sentiment, width, label='Without sentiment analysis')
    rects2 = ax.bar(x + width / 2, with_sentiment, width, label='With sentiment analysis')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Accuracy (%)')
    ax.set_title("Accuracy with vs without sentiment analysis")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    def autolabel(rects):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}%'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    autolabel(rects1)
    autolabel(rects2)

    fig.tight_layout()

    file_path = f'{path}/sentiment_vs_no_sentiment.png'
    plt.savefig(file_path, bbox_inches='tight', dpi=400)


def regularization(data):
    x = sorted(list(data.keys()))
    train = []
    val = []
    for key in x:
        train.append(data[key]['train_acc'])
        val.append(data[key]['test_acc'])

    plt.plot(x, train, '-o')
    plt.plot(x, val, '-o')
    plt.title("Variação da accuracy de acordo com o factor de regularização")
    plt.ylabel('accuracy')
    plt.xlabel('regularization factor')
    plt.legend(['Train accuracy', 'Validation accuracy'], loc='upper right')
    plt.savefig('graphics/r_Factor.png', bbox_inches='tight', dpi=400)
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

    number_classes = 5
    if number_classes:
        X, y = read_data(f"data/pruned_{number_classes}_entries.tsv")
    else:
        X, y = read_data()

    all_data_distribution(y, "graphics", "all_distribution.png")
    all_data_distribution(y, "graphics", "all_distribution_cv.png", with_cv_data=True)


if __name__ == '__main__':
    main()
