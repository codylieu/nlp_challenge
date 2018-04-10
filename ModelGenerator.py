import constants

import time

import numpy

# gensim modules
from gensim.models import Doc2Vec
from gensim.models.doc2vec import LabeledSentence

from SentenceVectorizer import SentenceVectorizer

""" Class that generates models for sentiment analysis using gensim's Doc2Vec """
class ModelGenerator():
  FILE_NAME_KEY = 'file_name'
  LABELED_SENTENCES_KEY = 'labeled_sentences'

  # Takes in file where each line represents a review and turns it into a LabeledSentence
  def __read_file_into_labeled_sentences(self, data_filepath, tag_prefix):
    data_file = open(data_filepath)

    # File structured as one review per line. Convert each file to List[List[String]].
    # Elements of outer list represent each line, elements of inner list represent each word in a sentence.
    data_content = [x.strip().split() for x in data_file.readlines()]

    # Convert content into Doc2Vec's LabeledSentence class to feed into model.
    data_as_labeled_sentences = [LabeledSentence(content, ['%s_%d' % (tag_prefix, index)]) for index, content in enumerate(data_content)]

    return data_as_labeled_sentences

  # Get list of vectors keyed to tag as a prefix
  def __get_vectors_list(self, tag):
    data_size = len(self.data_source_map[tag][self.LABELED_SENTENCES_KEY])

    vectors = []
    for x in range(0, data_size):
      vectors.append(self.model["%s_%d" % (tag, x)])

    return vectors

  # Classifier is a model from sklearn such as LogisticRegression which has fit and score methods
  # ex. http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
  def __generate(self):
    # Generate full set of LabeledSentences
    full_labeled_sentences = []
    for tag, metadata_map in self.data_source_map.iteritems():
      full_labeled_sentences.extend(metadata_map[self.LABELED_SENTENCES_KEY])

    # Generate the model
    print('Instantiating Doc2Vec model...')
    self.model = Doc2Vec(documents = full_labeled_sentences, min_count = 10, size = 100, workers = 1, sample= 1e-4, negative = 5)      

    # Save model with metadata such as current epoch time in filename for later processing
    # TODO: add in a utility to load historical models for comparison
    model_file_name = "review_model_%d.d2v" % int(round(time.time() * 1000))
    save_path = constants.GENERATED_MODEL_OUTPUT_DIR + model_file_name

    print('Finished model generation. Saving model to %s' % save_path)
    self.model.save(save_path)

    print('Finished model generation. Begin fitting classifier...')

    print('Constructing training vectors')
    positive_training_vectors = self.__get_vectors_list(constants.POSITIVE_TRAINING_TAG)
    negative_training_vectors = self.__get_vectors_list(constants.NEGATIVE_TRAINING_TAG)
    full_training_vectors = positive_training_vectors + negative_training_vectors

    print('Constructing training labels')
    positive_training_labels = numpy.ones(shape = len(positive_training_vectors))
    negative_training_labels = numpy.zeros(shape = len(negative_training_vectors))
    full_training_labels = numpy.concatenate((positive_training_labels, negative_training_labels), axis = 0)

    print('Fitting classifier to training data')
    self.classifier.fit(full_training_vectors, full_training_labels)

    print('Finished fitting classifier. Begin scoring classifier...')

    print('Constructing testing vectors')
    positive_testing_vectors = self.__get_vectors_list(constants.POSITIVE_TESTING_TAG)
    negative_testing_vectors = self.__get_vectors_list(constants.NEGATIVE_TESTING_TAG)
    full_testing_vectors = positive_testing_vectors + negative_testing_vectors

    print('Constructing testing labels')
    positive_testing_labels = numpy.ones(shape = len(positive_testing_vectors))
    negative_testing_labels = numpy.zeros(shape = len(negative_testing_vectors))
    full_testing_labels = numpy.concatenate((positive_testing_labels, negative_testing_labels), axis = 0)

    print('Scoring classifier')
    score = self.classifier.score(full_testing_vectors, full_testing_labels)

    print('Classifier received a score of %.4f' % score)

  def __init__(self, classifier):
    print('Initializing ModelGenerator...')
    self.classifier = classifier

    # Mapping of model tags to file name data is stored at and other metadata.
    # Tags are referenced later for model training and evaluation.
    self.data_source_map = {}

    self.data_source_map[constants.POSITIVE_TRAINING_TAG] = {
      self.FILE_NAME_KEY: constants.POSITIVE_TRAINING_FILE_NAME
    }
    self.data_source_map[constants.POSITIVE_TESTING_TAG] = {
      self.FILE_NAME_KEY: constants.POSITIVE_TESTING_FILE_NAME
    }
    self.data_source_map[constants.NEGATIVE_TRAINING_TAG] = {
      self.FILE_NAME_KEY: constants.NEGATIVE_TRAINING_FILE_NAME
    }
    self.data_source_map[constants.NEGATIVE_TESTING_TAG] = {
      self.FILE_NAME_KEY: constants.NEGATIVE_TESTING_FILE_NAME
    }

    # Iterate over map and store labeled_sentences along with file_name
    for tag, metadata_map in self.data_source_map.iteritems():
      file_path = constants.MODEL_GENERATION_DATA_DIR + metadata_map[self.FILE_NAME_KEY]
      # Transform the data stored at file_path into gensim's LabeledSentence class
      labeled_sentences = self.__read_file_into_labeled_sentences(file_path, tag)
      # Store the labeled_sentences in the map for later reference
      metadata_map[self.LABELED_SENTENCES_KEY] = labeled_sentences

    self.__generate()

    self.sentence_vectorizer = SentenceVectorizer(self.model)

  # Thin wrapper over sklearn regression predict methods.
  # Vectorizes input string using [[SentenceVectorizer]] and then predicts sentiment
  def predict(self, review):
    vector = self.sentence_vectorizer.naive_to_vector(review)
    return self.classifier.predict([vector])




