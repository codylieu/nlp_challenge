import constants

class TrainTestDataSplitter(object):

  def __train_test_split(self, file_path, training_ratio):
    file = open(file_path)
    content_as_list = [x.strip() for x in file.readlines()]

    data_size = len(content_as_list)
    print('Begin splitting data size of %d according to ratio %.2f' % (data_size, training_ratio))

    training_size = int(training_ratio * data_size)
    print('Training data size: %d, testing data size: %d' % (training_size, data_size - training_size))

    # TODO: maybe throw in some randomization
    training_data = content_as_list[:training_size]
    testing_data = content_as_list[training_size:]

    return training_data, testing_data

  def __write_data_to_file(self, data, file_name):
    output_file = open(constants.MODEL_GENERATION_DATA_DIR + file_name, 'w')

    for review in data:
      output_file.write("%s\n" % review)


  """ Splits the files found at positive_training_data_filepath and
  negative_training_data_filepath into training and test sets according to
  training_ratio and then writes to local filesystem """
  def split(self, positive_training_data_file_path, negative_training_data_file_path, training_ratio):
    assert training_ratio > 0 and training_ratio < 1, 'training_ratio must be between 0 and 1'

    print('Begin splitting %s' % positive_training_data_file_path)
    positive_training_data, positive_testing_data = self.__train_test_split(positive_training_data_file_path, training_ratio)

    print('Begin splitting %s' % negative_training_data_file_path)
    negative_training_data, negative_testing_data = self.__train_test_split(negative_training_data_file_path, training_ratio)

    print('Done splitting, begin saving files to %s' % constants.MODEL_GENERATION_DATA_DIR)

    print('Writing %s' % constants.POSITIVE_TRAINING_FILE_NAME)
    self.__write_data_to_file(positive_training_data, constants.POSITIVE_TRAINING_FILE_NAME)

    print('Writing %s' % constants.POSITIVE_TESTING_FILE_NAME)
    self.__write_data_to_file(positive_testing_data, constants.POSITIVE_TESTING_FILE_NAME)

    print('Writing %s' % constants.NEGATIVE_TRAINING_FILE_NAME)
    self.__write_data_to_file(negative_training_data, constants.NEGATIVE_TRAINING_FILE_NAME)

    print('Writing %s' % constants.NEGATIVE_TESTING_FILE_NAME)
    self.__write_data_to_file(negative_testing_data, constants.NEGATIVE_TESTING_FILE_NAME)





