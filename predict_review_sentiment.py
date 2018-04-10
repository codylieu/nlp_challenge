from TrainTestDataSplitter import TrainTestDataSplitter

from ModelGenerator import ModelGenerator
from sklearn.linear_model import LogisticRegression

def main():
    # Split original data files into training and test sets
    splitter = TrainTestDataSplitter()
    splitter.split('data/original/positive_reviews.txt', 'data/original/negative_reviews.txt', .5)

    # Can switch out the classifier passed to ModelGenerator
    classifier = LogisticRegression()
    modelGenerator = ModelGenerator(classifier)

    while True:
      review = raw_input('Enter movie review to predict sentiment:\n')
      if len(review) == 0:
        break
      else:
        # With my presets, prediction only returns 0 or 1.
        # Linear regression gives more interesting results.
        prediction = modelGenerator.predict(review)
        print("Review sentiment: %.4f" % prediction)

if __name__ == '__main__':
    main()