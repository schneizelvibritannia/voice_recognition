from __future__ import unicode_literals, print_function
import logging
import os
import json
import random
import datetime
import uuid
from pathlib import Path
from dotenv import load_dotenv

from sutime import SUTime
from fuzzywuzzy import process
from word2number import w2n
from collections import defaultdict

import spacy
from spacy.util import minibatch
from spacy.scorer import Scorer

load_dotenv()


def convert_dataset_json(input_file=None, output_file=None, format="spacy"):
    """
    Convert annotations from label studio to required format.
    :param input_file:
    :param output_file:
    :param format:
    :return:
    """
    if format == 'spacy':
        try:
            with open(input_file, 'r') as f:
                data = json.load(f)
            data_dict = {
                "annotations": []
            }
            classes = []
            for item in data:
                text = item["text"].lower()
                annotations = item["label"]
                annotation = []
                for annotation_item in annotations:
                    start = annotation_item["start"]
                    end = annotation_item["end"]
                    label = annotation_item["labels"][0].upper()
                    if label not in classes:
                        classes.append(label)
                    annotation.append([start, end, label])
                entities = {
                    "entities": annotation
                }
                data_dict["annotations"].append((text, entities))
            data_dict["classes"] = classes

            with open(output_file, 'w') as fp:
                json.dump(data_dict, fp, indent=4, sort_keys=True)

        except Exception as e:
            logging.exception("Unable to process " + input_file + "\n" + "error = " + str(e))
            return None


class SpaceStatusReportNER:
    def __init__(self, model_path=None):
        """
            Class for processing Status Report queries from space using NLP.

        :param model_path: Path to NLP model
        """
        jar_files_dir = os.environ.get('SUTIME_PATH')
        if model_path is None:
            model_path = os.path.join('./models',
                                      'model_' + str(uuid.uuid4()) + '_' + str(datetime.datetime.today().date()))
            print('Default model path set to : "{}"'.format(model_path))
        self.model_path = model_path
        try:
            self.nlp = spacy.load(model_path)
            self.ner_labels = list(self.nlp.pipe_labels.values())[0]

        except OSError:
            self.nlp = None
            print('No saved models found in given path, use train function to save model.')
        jar_path = None
        if jar_files_dir:
            jar_path = os.path.dirname(jar_files_dir)
        self.sutime = SUTime(jars=jar_path, mark_time_ranges=True, include_range=True)

    @staticmethod
    def load_data(data, test_train_split=0.2):
        """

        :param data: []
        :param test_train_split: float [0,1]
        :return:test_set, train_set
        """
        count = len(data)
        assert count > 0
        assert 0 <= test_train_split <= 1
        split = int(count * test_train_split)
        random.shuffle(data)
        return data[0:split], data[split:]

    def train(self, model=None, new_model_name='new_model', output_dir=None,
              n_iter=25, dataset_path=None, test_train_split=0.2, dropout_rate=0.2):
        """

        :param model: Pretrained model path
        :param new_model_name:
        :param output_dir: Path to save model artifacts
        :param n_iter: Iterations
        :param dataset_path: Path to JSON file with data.
        :param test_train_split: percentage of data split for test and train
        :param dropout_rate: Rate of dropout used in model
        :return:
        """

        assert dataset_path
        if not output_dir:
            output_dir = self.model_path

        # Loading training data
        with open(dataset_path, 'rb') as fp:
            training_data = json.load(fp)

        label = training_data["classes"]
        training_data = training_data["annotations"]

        """Setting up the pipeline and entity recognizer, and training the new entity."""

        if model is not None:
            nlp = spacy.load(model)  # load existing spacy model
            print("Loaded model '%s'" % model)
        else:
            # nlp = spacy.blank('en')  # create blank Language class
            nlp = spacy.load('en_core_web_md')
            print("Created blank 'en' model")
        if 'ner' not in nlp.pipe_names:
            # ner = nlp.create_pipe('ner')
            nlp.add_pipe('ner')
            ner = nlp.get_pipe('ner')
        else:
            ner = nlp.get_pipe('ner')

        for i in label:
            ner.add_label(i)  # Add new entity labels to entity recognizer

        # optimizer = nlp.begin_training()
        optimizer = nlp.create_optimizer()

        scorer = Scorer(nlp)

        # Get names of other pipes to disable them during training to train only NER
        other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']
        with nlp.disable_pipes(*other_pipes):  # only train NER
            for itn in range(n_iter):
                test_data, train_data = self.load_data(training_data, test_train_split=test_train_split)
                losses = {}
                batches = minibatch(train_data, size=5)
                examples = []
                for batch in batches:
                    for text, annotations in batch:
                        doc = nlp.make_doc(text)
                        example = spacy.training.example.Example.from_dict(doc, annotations)
                        examples.append(example)

                        nlp.update([example], sgd=optimizer, drop=dropout_rate,
                                   losses=losses)
                print('Losses', losses)

                try:
                    preds = []
                    for input_, annot in train_data:
                        pred_value = spacy.training.example.Example.from_dict(nlp(input_), annot)
                        preds.append(pred_value)
                except Exception as e:
                    print(e)
                score = scorer.score(preds)
                print('Train Scores', score)

                try:
                    preds = []
                    for input_, annot in test_data:
                        pred_value = spacy.training.example.Example.from_dict(nlp(input_), annot)
                        preds.append(pred_value)
                except Exception as e:
                    print(e)
                score = scorer.score(preds)
                print('Test Scores', score)

        # Test the trained model
        test_text = random.sample(training_data, 1)[0][0]
        print(test_text)
        doc = nlp(test_text)
        print("Entities in '%s'" % test_text)
        for ent in doc.ents:
            print(ent.label_, ent.text)

        # Save model
        if output_dir is None:
            output_dir = self.model_path
        output_dir = Path(output_dir)
        if not output_dir.exists():
            output_dir.mkdir()
        nlp.meta['name'] = new_model_name  # rename model
        nlp.to_disk(output_dir)
        print("Saved model to", output_dir)

        # Test the saved model
        print("Loading from", output_dir)
        nlp2 = spacy.load(output_dir)

        if self.nlp is None:
            self.nlp = nlp2
            self.ner_labels = list(self.nlp.pipe_labels.values())[0]

        print(training_data)
        test_text = random.sample(training_data, 1)[0][0]
        print(test_text)
        doc2 = nlp2(test_text)
        print("Entities in '%s'" % test_text)
        for ent in doc2.ents:
            print(ent.label_, ent.text)

    @staticmethod
    def get_match_from_list(input_text, input_list):
        match = process.extract(input_text, input_list)
        return match[0][0]

    def get_date_from_text(self, text):
        try:
            status_date_str = self.sutime.parse(text)[0]['value']
            status_date = datetime.datetime.strptime(status_date_str, '%Y-%m-%d').date()
            today = datetime.datetime.today().date()
            date_past_week = today - datetime.timedelta(days=7)
            if status_date < date_past_week or status_date > today:
                return str(today)
            return status_date_str
        except Exception:
            return str(datetime.datetime.today().date())

    def get_entities(self, input_text, beam_width=16, beam_density=0.0001):

        input_text = input_text.lower()
        entities = self.nlp(input_text)

        # Number of alternate analyses to consider. More is slower, and not necessarily better
        # -- you need to experiment on your problem.
        # print(beam_width)
        # This clips solutions at each step. We multiply the score of the top-ranked action by this value,
        # and use the result as a threshold. This prevents the parser from exploring options that look very unlikely,
        # saving a bit of efficiency. Accuracy may also improve, because we've trained on greedy objective.
        # print(beam_density)
        beams = self.nlp.get_pipe("ner").beam_parse([entities], beam_width, beam_density)

        entity_scores = defaultdict(float)
        for beam in beams:
            for score, ents in self.nlp.get_pipe("ner").moves.get_beam_parses(beam):
                for start, end, label in ents:
                    entity_scores[label] += score

        return entities, entity_scores

    @staticmethod
    def get_time_from_text(text):
        texts = text.split(' ')
        for t in texts:
            try:
                hrs = w2n.word_to_num(t)
                if hrs > 16:
                    hrs = 16
                return hrs
            except ValueError:
                continue
        return 8

    def process(self, input_text, validate=False, projects=[]):

        entities, scores = self.get_entities(input_text)
        data = {}
        entity_list = entities.ents
        status = []
        for e in entity_list:
            label = e.label_
            value = e.text

            if validate:
                if label == 'PROJECT':
                    if len(projects) > 0:
                        value = self.get_match_from_list(value, projects)
                        if not value:
                            # value = 'N/A'
                            value = projects[0]
                if label == 'DATE':
                    value = self.get_date_from_text(value)
                if label == 'TIME SPENT':
                    value = self.get_time_from_text(value)
            if label == 'STATUS':
                status.append(value)
                value = ' '.join(status)
            data[label] = value
            # data[label + '_SCORE'] = scores[label]

        if validate:
            labels_list = [et.label_ for et in entity_list]
            if 'PROJECT' not in labels_list:
                if projects:
                    data['PROJECT'] = projects[0]
                else:
                    data['PROJECT'] = 'N/A'
            if 'DATE' not in labels_list:
                data['DATE'] = self.get_date_from_text('today')
            if 'TIME SPENT' not in labels_list:
                data['TIME SPENT'] = 8
            # if 'STATUS' not in labels_list:
            #     data['STATUS'] = 'Worked on project ' + data['PROJECT']
        return data
