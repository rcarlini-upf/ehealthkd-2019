# coding: utf8

import argparse
import re
from pathlib import Path

import spacy
import pandas as pd

from collections import Counter

from sklearn.metrics import precision_score, recall_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import DictVectorizer
from sklearn import svm

from scripts.utils import Collection, Keyphrase, Relation, Sentence


class BaselineClassifier:

    def __init__(self):
        self.keyphrases = {}
        self.relations = {}

    def train(self, finput):

        collection = Collection()
        collection.load(finput)

        self.keyphrases.clear()
        for sentence in collection.sentences:
            for keyphrase in sentence.keyphrases:
                text = keyphrase.text.lower()
                self.keyphrases[text] = keyphrase.label

        self.relations.clear()
        for sentence in collection.sentences:
            for relation in sentence.relations:
                origin = relation.from_phrase
                origin_text = origin.text.lower()
                destination = relation.to_phrase
                destination_text = destination.text.lower()

                self.relations[origin_text, origin.label, destination_text, destination.label] = relation.label

    def predict_entities(self, collection):

        next_id = 0
        for instance_keyphrase, label in self.keyphrases.items():
            for sentence in collection.sentences:
                text = sentence.text.lower()
                pattern = r'\b' + instance_keyphrase + r'\b'
                for match in re.finditer(pattern, text):
                    keyphrase = Keyphrase(sentence, label, next_id, [match.span()])
                    keyphrase.split()
                    next_id += 1

                    sentence.keyphrases.append(keyphrase)

    def predict_relations(self, sentence):

        for origin in sentence.keyphrases:
            origin_text = origin.text.lower()
            for destination in sentence.keyphrases:
                destination_text = destination.text.lower()
                try:
                    label = self.relations[origin_text, origin.label, destination_text, destination.label]
                except KeyError:
                    continue
                relation = Relation(sentence, origin.id, destination.id, label)
                sentence.relations.append(relation)

    def test(self, finput: Path, skip_A, skip_B):
        collection = Collection()

        if skip_A:
            collection.load_keyphrases(finput)
        else:
            collection.load_input(finput)
            self.predict_entities(collection)

        if not skip_B:
            for sentence in collection.sentences:
                self.predict_relations(sentence)
                sentence.remove_dup_relations()

        return collection


class InitialClassifier():

    def __init__(self, clf_instance=None):

        if clf_instance is None:
            self.clf = svm.SVC(gamma='scale')
        else:
            self.clf = clf_instance

        self.concept_vectorizer = None
        self.feature_vectorizer = None

    def preprocess(self, collection, train=False):
        process_ES = spacy.load('es_core_news_sm')
        accepted_pos = ('NOUN', 'PROPN')
        accepted_pos = ()

        df = pd.DataFrame(columns=['concept', 'pos', 'pre_pos', 'post_pos', 'label'])

        pos_list = []
        for sentence in collection.sentences:
            doc = process_ES(sentence.text)

            for token in doc:

                if not accepted_pos or token.pos_ in accepted_pos:

                    start = token.idx
                    end = start + len(token)
                    keyphrase = sentence.find_keyphrase(start=start, end=end)

                    if keyphrase:
                        # label = keyphrase.label
                        label = 1
                        pos_list.append(token.pos_)
                    else:
                        label = 0

                    pre_pos = 'START'
                    if token.i-1 >= 0:
                        pre_pos = str(doc[token.i-1])

                    post_pos = 'END'
                    if token.i+1 < len(doc):
                        post_pos = str(doc[token.i-1])

                    features = {
                        'concept': token.text,
                        'pos': token.pos_,
                        'pre_pos': pre_pos,
                        'post_pos': post_pos,
                        'label': label}
                    df = df.append(features, ignore_index=True)

        print('Counter pos: %s' % Counter(pos_list))

        no_label_dict = df.T.drop(['label']).T.to_dict('records')
        if train:
            self.concept_vectorizer = CountVectorizer()
            self.concept_vectorizer.fit(df.concept)

            self.feature_vectorizer = DictVectorizer()
            self.feature_vectorizer.fit(no_label_dict)

        vectorized_data = self.feature_vectorizer.transform(no_label_dict)

        return vectorized_data, df.label.values.astype('int')

    def train(self, finput):

        collection = Collection()
        collection.load(finput)

        """
        full_text = ""
        for sentence in collection.sentences:
            full_text += sentence.text
            full_text += "\n"

        doc = es_pipeline.nlp(full_text)
        """

        x_train, y_train = self.preprocess(collection, True)
        print('Counter train: %s' % Counter(y_train))

        fit_result = self.clf.fit(x_train, y_train)
        print("Success at training!")

        return fit_result

    def predict_entities(self, collection):

        if self.concept_vectorizer is None:
            raise Exception("Not trained yet!")

        X_test, y_test = self.preprocess(collection)
        print("Accuracy: " + str(self.clf.score(X_test, y_test)))

        y_pred = self.clf.predict(X_test)

        print('Counter test: %s' % Counter(y_test))
        print('Counter pred: %s' % Counter(y_pred))
        print('Average precision: {0:0.2f}'.format(precision_score(y_test, y_pred)))
        print('Average recall: {0:0.2f}'.format(recall_score(y_test, y_pred)))

    def predict_relations(self, sentence):
        pass

    def test(self, finput: Path, skip_A=False, skip_B=False):

        collection = Collection()
        collection.load(finput)

        self.predict_entities(collection)

        return collection


def main_baseline(training_input, testing_input, foutput, skip_A, skip_B, model_path=None):

    model = BaselineClassifier()
    model.train(training_input)  # train(training_input)

    collection = model.test(testing_input, skip_A, skip_B)
    collection.dump(foutput, skip_empty_sentences=False)


def main_initial(training_input, testing_input, foutput, skip_A, skip_B, model_path=None):

    if model_path:
        model = InitialClassifier(model_path)
    else:
        model = InitialClassifier()
        model.train(training_input)

    doc = model.test(testing_input, skip_A, skip_B)
    # doc.dump(foutput, skip_empty_sentences=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('training')
    parser.add_argument('test')
    parser.add_argument('output')
    parser.add_argument('-m', '--model')
    parser.add_argument('--skip-A', action='store_true')
    parser.add_argument('--skip-B', action='store_true')
    args = parser.parse_args()

    model_path = None
    if args.model:
        model_path = Path(args.model)

    main_initial(Path(args.training), Path(args.test), Path(args.output), args.skip_A, args.skip_B, model_path=model_path)
