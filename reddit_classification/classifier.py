import string
from nltk import download, WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from tensorflow.keras import Sequential, layers
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from .utils import read_model, save_model, prune_vocabulary_until_normalized, save_logs


download('punkt')
download('stopwords')
download('wordnet')


def tokenize(text, language='english'):
    tokens = word_tokenize(text)
    tokens = [w.lower() for w in tokens]
    table = str.maketrans('', '', string.punctuation)
    words = [w.translate(table) for w in tokens]
    stop_words = set(stopwords.words(language))

    lem = WordNetLemmatizer()
    words = [lem.lemmatize(w, pos="v") for w in words if w not in stop_words and not w.isdigit() and len(w) > 2]

    return ' '.join(words)


def convert_input(x_train, x_test):
    tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=10)

    x_train_v = tfidf.fit_transform(x_train)
    terms = tfidf.get_feature_names()

    limit = 100

    pruned_terms_index, _, _ = prune_vocabulary_until_normalized(x_train_v, terms, limit=limit)
    pruned_terms = [terms[term_index] for term_index in pruned_terms_index]

    tf = CountVectorizer(vocabulary=pruned_terms)

    x_train = tf.fit_transform(x_train).todense()
    x_test = tf.transform(x_test).todense()

    return x_train, x_test, tf


def shuffle_split_data(x, y, test_size=0.25, shuffle=True):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=42, shuffle=shuffle)
    return x_train, x_test, y_train, y_test


def create_model(input_dim):
    model = Sequential()
    model.add(layers.Dense(100, activation="relu", input_dim=input_dim))
    model.add(layers.Dense(100, activation="relu"))
    model.add(layers.Dense(100, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model


def pick_best_model(identifier, x, y, output_path, epochs=50, batch_size=200, save_logs_status=True):
    x = [tokenize(text) for text in x]

    x_train, x_test, y_train, y_test = shuffle_split_data(x, y, 0.2, True)
    x_train, x_test, tf = convert_input(x_train, x_test)
    vocab_size = len(tf.get_feature_names())

    model = KerasClassifier(build_fn=create_model, epochs=epochs, batch_size=batch_size, verbose=True,
                            input_dim=vocab_size, validation_split=0.2)

    history = model.fit(x_train, y_train)

    config = {
        'epochs': epochs,
        'batch_size': batch_size,
        'max_len': vocab_size
    }

    if save_logs_status:
        data = {
            'data_size': len(x),
            'train_acc': model.score(x_train, y_train),
            'test_acc': model.score(x_test, y_test)
        }
        save_logs(identifier, data)

    save_model(identifier, config, model, tf, output_path)

    return config, model, tf


def predict(identifier, x):
    config, model, tf = read_model(identifier)

    classifier = KerasClassifier(build_fn=create_model, epochs=config['epochs'], batch_size=config['batch_size'])

    classifier.model = model
    x = [tokenize(text) for text in x]
    x = tf.transform(x).todense()

    return model.predict(x)
