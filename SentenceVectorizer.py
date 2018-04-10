import numpy

from gensim.models import Doc2Vec

""" Interface for converting a sentence to a vector using a Doc2Vec model's word vectors """
class SentenceVectorizer(object):
  def __init__(self, model):
    self.model = model

  # naive implementation which adds the vector representation of all the words together and then divides by the square root of the dot product
  def naive_to_vector(self, sentence):
    feature_dimension_size = self.model[0].size
    vectorized_sentence = numpy.zeros(feature_dimension_size)

    num_words = 0
    sentence_as_list = sentence.strip().split()
    for word in sentence_as_list:
      try:
        vectorized_sentence = numpy.add(vectorized_sentence, self.model[word])
        num_words += 1
      except:
        pass
    return vectorized_sentence / numpy.sqrt(vectorized_sentence.dot(vectorized_sentence))