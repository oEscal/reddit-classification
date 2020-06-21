import string
import json

from nltk import download, WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from keras.preprocessing.sequence import pad_sequences
from keras import Sequential, layers
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from reddit_classification.utils import read_model, save_model, prune_vocabulary_until_normalized

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
    words = [lemmatizer.lemmatize(w, pos="v") for w in words if w not in stop_words and not w.isdigit() and len(w) > 2]

    return ' '.join(words)


def convert_input(x_train, x_test):
    tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=5)

    x_train_v = tfidf.fit_transform(x_train)
    terms = tfidf.get_feature_names()

    pruned_terms_index, _, _ = prune_vocabulary_until_normalized(x_train_v, terms, limit=200)
    pruned_terms = [terms[term_index] for term_index in pruned_terms_index]

    tf = CountVectorizer(vocabulary=pruned_terms)

    x_train = tf.fit_transform(x_train).todense()
    x_test = tf.transform(x_test).todense()

    return x_train, x_test, len(tf.get_feature_names()), tf


def shuffle_split_data(x, y, test_size=0.25, shuffle=True):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=42, shuffle=shuffle)
    return x_train, x_test, y_train, y_test


def create_model(input_dim):
    model = Sequential()
    model.add(layers.Dense(100, activation="relu", input_dim=input_dim))
    model.add(layers.Dense(50, activation="relu"))
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer='nadam', loss='binary_crossentropy', metrics=['accuracy'])

    return model


def pick_best_model(identifier, x, y, param_grid=None, epochs=200, n_jobs=-1,
                    batch_size=50, save_logs=True):
    x = [tokenize(text) for text in x]

    x_train, x_test, y_train, y_test = shuffle_split_data(x, y, 0.2, True)
    x_train, x_test, vocab_size, tokenizer = convert_input(x_train, x_test)

    if param_grid is None:
        param_grid = {
            "input_dim": [x_train.shape[1]]
        }

        model = KerasClassifier(build_fn=create_model, epochs=epochs, batch_size=batch_size, verbose=True)

        grid = RandomizedSearchCV(estimator=model, param_distributions=param_grid, cv=min(4, len(y_train)), verbose=2,
                                  n_iter=1, n_jobs=n_jobs)

        grid_result = grid.fit(x_train, y_train)

        best_model = grid_result.best_estimator_

        #plt.plot(best_model.model.history['acc'])
        #plt.plot(best_model.model.history['val_acc'])
        #plt.title('model accuracy')
        #plt.ylabel('accuracy')
        #plt.xlabel('epoch')
        #plt.legend(['train', 'val'], loc='upper left')
        #plt.show()

        config = {
            'epochs': epochs,
            'batch_size': batch_size,
            'max_len': x_train.shape[1],
            'padding': 'post'
        }

        if save_logs:
            logs_file = f'logs/{identifier}.logs'
            with open(logs_file, 'w') as f:
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
