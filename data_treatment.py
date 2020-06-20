import csv
import random


def read_data(file_path='data/entries.tsv'):
    x, y = [], []

    with open(file_path) as f:
        rows = list(csv.reader(f, delimiter='\t', quotechar='"'))
        for row in rows[1:]:
            theme, text = row
            x.append(text)
            y.append(theme)

    return x, y


def prune_data(x, y, n_classes=20):
    all_classes = list(set(y))
    n_classes = min(n_classes, len(all_classes))
    picked_classes = set()

    for _ in range(n_classes):
        picked_classes.add(all_classes.pop(random.randint(0, len(all_classes) - 1)))

    x_prune, y_prune = [], []

    for i in range(len(y)):
        if y[i] in picked_classes:
            x_prune.append(x[i])
            y_prune.append(y[i])

    return x_prune, y_prune


def write_data(x, y, file_path):
    with open(file_path, 'w') as f:
        tsv_writer = csv.writer(f, delimiter='\t', quotechar='"')
        tsv_writer.writerow(['theme', 'text'])
        for i in range(len(y)):
            tsv_writer.writerow([y[i], x[i]])


"""
x, y = read_data("data/entries.tsv")
x, y = prune_data(x, y, 5)
write_data(x, y, "data/pruned_5_entries.tsv")
"""
