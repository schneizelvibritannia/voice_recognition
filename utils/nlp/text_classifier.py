import os
import pickle
import uuid
from datetime import datetime

import pandas as pd
import spacy
import random
from spacy.util import minibatch, compounding
from spacy.training.example import Example
from spacy.pipeline.textcat import single_label_cnn_config, Config


class TextClassifier:
    def __init__(self, model_path=None):
        """
        Class for classifying input query from space using spacy
        classes : status report, vacation request
        :param model_path:
        """
        if model_path is None:
            model_path = os.path.join('./models',
                                      'model_' + str(uuid.uuid4()) + '_' + str(datetime.datetime.today().date()))
            print('Default model path set to : "{}"'.format(model_path))
        self.model_path = model_path
        try:
            self.nlp = spacy.load(self.model_path)
        except OSError:
            self.nlp = None
            print('No saved models found in given path, use train function to save model.')

    @staticmethod
    def preprocess_data(dataset_path=None, model_name=None):
        """

        :param dataset_path: dataset to be trained
        :param model_name: spacy model name to be loaded
        :return:
        """
        assert dataset_path
        data = pd.read_csv(dataset_path)

        if model_name is not None:
            nlp = spacy.load(model_name)
        else:
            nlp = spacy.load("en_core_web_md")

        config = Config().from_str(single_label_cnn_config)
        if "textcat" not in nlp.pipe_names:
            nlp.add_pipe('textcat', config=config, last=True)
        text_cat = nlp.get_pipe('textcat')

        text_cat.add_label("Status Report")
        text_cat.add_label("Vacation Request")

        data["binary"] = data.apply(lambda row: 1 if row['type_request'] == 'status' else 0, axis=1)
        data['tuples'] = data.apply(lambda row: (row['request_description'], row['binary']), axis=1)
        train = data['tuples'].tolist()
        return train, nlp, text_cat

    @staticmethod
    def load_data(train_data, limit=0, split=0.8):
        """

        :param train_data: Data to be trained
        :param limit:
        :param split:
        :return:
        """
        # Shuffle the data
        random.shuffle(train_data)
        texts, labels = zip(*train_data)
        # get the categories for each review
        cats = [{"Status Report": bool(y), "Vacation Request": not bool(y)} for y in labels]

        # Splitting the training and evaluation data
        split = int(len(train_data) * split)
        return (texts[:split], cats[:split]), (texts[split:], cats[split:])

    def training(self, train, nlp, text_cat, n_iter=10, dropout_rate=0.35):
        """

        :param nlp:
        :param train: train data
        :param text_cat:
        :param n_iter: Iterations
        :param dropout_rate: Rate of dropout used in model
        :return:
        """
        n_texts = 23486
        # Calling the load_data() function
        (train_texts, train_cats), (dev_texts, dev_cats) = self.load_data(train, limit=n_texts)
        # Processing the final format of training data
        train_data = list(zip(train_texts, [{'cats': cats} for cats in train_cats]))
        other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'textcat']
        with nlp.disable_pipes(*other_pipes):  # only train textcat
            optimizer = nlp.initialize()
        print("Training the model...")
        print('{:^5}\t{:^5}\t{:^5}\t{:^5}'.format('LOSS', 'P', 'R', 'F'))
        # Performing training
        for i in range(n_iter):
            losses = {}
            batches = minibatch(train_data, size=compounding(4., 32., 1.001))
            for batch in batches:
                for text, annotations in batch:
                    # texts, annotations = zip(*batch)
                    doc = nlp.make_doc(text)
                    example = Example.from_dict(doc, annotations)
                    # Update the model
                    nlp.update([example], sgd=optimizer, losses=losses, drop=dropout_rate)
                    # nlp.update(texts, annotations, sgd=optimizer, drop=dropout_rate,losses=losses)

        # Calling the evaluate() function and printing the scores
        # with text_cat.model.use_params(optimizer.averages):
        scores = self.evaluate(nlp.tokenizer, text_cat, dev_texts, dev_cats)
        print('{0:.3f}\t{1:.3f}\t{2:.3f}\t{3:.3f}'
              .format(losses['textcat'], scores['textcat_p'],
                      scores['textcat_r'], scores['textcat_f']))
        return nlp

    @staticmethod
    def evaluate(tokenizer, textcat, texts, cats):
        """

        :param tokenizer:
        :param textcat:
        :param texts:
        :param cats:
        :return:
        """
        docs = (tokenizer(text) for text in texts)
        tp = 0.0  # True positives
        fp = 1e-8  # False positives
        fn = 1e-8  # False negatives
        tn = 0.0  # True negatives
        for i, doc in enumerate(textcat.pipe(docs)):
            gold = cats[i]
            for label, score in doc.cats.items():
                if label not in gold:
                    continue
                if label == "Vacation Request":
                    continue
                if score >= 0.5 and gold[label] >= 0.5:
                    tp += 1.0
                elif score >= 0.5 and gold[label] < 0.5:
                    fp += 1.0
                elif score < 0.5 and gold[label] < 0.5:
                    tn += 1
                elif score < 0.5 and gold[label] >= 0.5:
                    fn += 1
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        if (precision + recall) == 0:
            f_score = 0.0
        else:
            f_score = 2 * (precision * recall) / (precision + recall)
        print(precision, recall, f_score)
        return {"textcat_p": precision, "textcat_r": recall, "textcat_f": f_score}

    def train(self, dataset_path=None, model_name=None):
        """

        :param dataset_path:
        :param model_name:
        :return:
        """

        train_data, nlp, text_cat = self.preprocess_data(dataset_path, model_name)
        nlp_trained = self.training(train_data, nlp, text_cat)
        representations = {"nlp": nlp_trained}
        nlp_trained.to_disk(self.model_path)

    def process(self, input_text, threshold=0.6):

        """
        Process input text and get classification

        :param input_text:
        :param threshold: threshold to be classified as status report
        :return:
        """

        stop_words = ['leave', 'vacation', 'send', 'sent','absent','absence','off']
        try:
            doc = self.nlp(input_text)
            c = list(map(lambda x: x in input_text, stop_words))
            if doc.cats["Vacation Request"] > threshold:

                return {'class': "Vacation Request", 'confidence': doc.cats["Vacation Request"]}
            elif doc.cats["Status Report"] > threshold:
                if any(c):
                    return {'class': "Vacation Request", 'confidence': doc.cats["Vacation Request"]}
                return {'class': "Status Report", 'confidence': doc.cats["Status Report"]}
        except Exception as e:
            return None
