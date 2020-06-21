from reddit_classification.data_treatment import read_data
from reddit_classification.classifier import pick_best_model


def main():
    x, y = read_data('data/pruned_3_entries.tsv')
    identifier = "modelo_1"

    config, best_model, tokenizer = pick_best_model(identifier, x, y)


if __name__ == '__main__':
    main()
