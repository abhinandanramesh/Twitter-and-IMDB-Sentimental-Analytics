import sys
import collections
import sklearn.naive_bayes
import sklearn.linear_model
import nltk
import random
random.seed(0)
from gensim.models.doc2vec import LabeledSentence, Doc2Vec
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
#nltk.download("stopwords")          # Download the stop words from nltk


# User input path to the train-pos.txt, train-neg.txt, test-pos.txt, and test-neg.txt datasets
if len(sys.argv) != 3:
    print "python sentiment.py <path_to_data> <0|1>"
    print "0 = NLP, 1 = Doc2Vec"
    exit(1)
path_to_data = sys.argv[1]
method = int(sys.argv[2])



def main():
    train_pos, train_neg, test_pos, test_neg = load_data(path_to_data)
    
    if method == 0:
        train_pos_vec, train_neg_vec, test_pos_vec, test_neg_vec = feature_vecs_NLP(train_pos, train_neg, test_pos, test_neg)
        nb_model, lr_model = build_models_NLP(train_pos_vec, train_neg_vec)
    if method == 1:
        train_pos_vec, train_neg_vec, test_pos_vec, test_neg_vec = feature_vecs_DOC(train_pos, train_neg, test_pos, test_neg)
        nb_model, lr_model = build_models_DOC(train_pos_vec, train_neg_vec)
    print "Naive Bayes"
    print "-----------"
    evaluate_model(nb_model, test_pos_vec, test_neg_vec, True)
    print ""
    print "Logistic Regression"
    print "-------------------"
    evaluate_model(lr_model, test_pos_vec, test_neg_vec, True)



def load_data(path_to_dir):
    """
    Loads the train and test set into four different lists.
    """
    train_pos = []
    train_neg = []
    test_pos = []
    test_neg = []
    with open(path_to_dir+"train-pos.txt", "r") as f:
        for i,line in enumerate(f):
            words = [w.lower() for w in line.strip().split() if len(w)>=3]
            train_pos.append(words)
    with open(path_to_dir+"train-neg.txt", "r") as f:
        for line in f:
            words = [w.lower() for w in line.strip().split() if len(w)>=3]
            train_neg.append(words)
    with open(path_to_dir+"test-pos.txt", "r") as f:
        for line in f:
            words = [w.lower() for w in line.strip().split() if len(w)>=3]
            test_pos.append(words)
    with open(path_to_dir+"test-neg.txt", "r") as f:
        for line in f:
            words = [w.lower() for w in line.strip().split() if len(w)>=3]
            test_neg.append(words)

    return train_pos, train_neg, test_pos, test_neg



def feature_vecs_NLP(train_pos, train_neg, test_pos, test_neg):
    """
    Returns the feature vectors for all text in the train and test datasets.
    """
    # English stopwords from nltk
    stopwords = set(nltk.corpus.stopwords.words('english'))
    
    # Determine a list of words that will be used as features. 
    # This list should have the following properties:
    #   (1) Contains no stop words
    #   (2) Is in at least 1% of the positive texts or 1% of the negative texts
    #   (3) Is in at least twice as many postive texts as negative texts, or vice-versa.
    positive = {}    
    negative = {}
    
    for line in train_pos:
	for word in line:
		if word not in stopwords:
			positive.setdefault(word,0)
			positive[word] += 1

    for line in train_neg:
	for word in line:
		if word not in stopwords:
			negative.setdefault(word,0)
			negative[word] += 1

    feature = []
    for line in train_pos:
	for word in line:
		if word not in feature:
			if word in positive and word in negative:
				if (float(positive[word])/len(train_pos) >= .01 or float(negative[word])/len(train_neg) >= .01) and (positive[word] >= 2*negative[word] or negative[word] >= 2*positive[word]):
					feature.append(word)


    for line in train_neg:
	for word in line:
		if word not in feature:
			if word in positive and word in negative:
				if (float(positive[word])/float(len(train_pos)) >= .01 or float(negative[word])/float(len(train_neg)) >= .01) and (positive[word] >= 2*negative[word] or negative[word] >= 2*positive[word]):
					feature.append(word)

    # Using the above words as features, construct binary vectors for each text in the training and test set.
    # These should be python lists containing 0 and 1 integers.
    # YOUR CODE HERE
    train_pos_vec = []
    train_neg_vec = []
    test_pos_vec = []
    test_neg_vec = []

    for line in train_pos:
	binary = [0] * len(feature)
	for word in line:
		if word in feature:
			binary[feature.index(word)] = 1
	train_pos_vec.append(binary)

    for line in train_neg:
	binary = [0] * len(feature)
	for word in line:
		if word in feature:
			binary[feature.index(word)] = 1
	train_neg_vec.append(binary)

    for line in test_pos:
	binary = [0] * len(feature)
	for word in line:
		if word in feature:
			binary[feature.index(word)] = 1
	test_pos_vec.append(binary)

    for line in test_neg:
	binary = [0] * len(feature)
	for word in line:
		if word in feature:
			binary[feature.index(word)] = 1
	test_neg_vec.append(binary)

    # Return the four feature vectors
    return train_pos_vec, train_neg_vec, test_pos_vec, test_neg_vec



def feature_vecs_DOC(train_pos, train_neg, test_pos, test_neg):
    """
    Returns the feature vectors for all text in the train and test datasets.
    """
    # Doc2Vec requires LabeledSentence objects as input.
    # Turn the datasets from lists of words to lists of LabeledSentence objects.
    labeled_train_pos = []
    labeled_train_neg = []
    labeled_test_pos = []
    labeled_test_neg = []

    i = 0
    for line in train_pos:
	sentence = LabeledSentence(line, ['TRAIN_POS_%s' % str(i)])
	labeled_train_pos.append(sentence)
        i += 1

    i = 0    
    for line in train_neg:
	sentence = LabeledSentence(line, ['TRAIN_NEG_%s' % str(i)])
	labeled_train_neg.append(sentence)
        i += 1

    i = 0    
    for line in test_pos:
	sentence = LabeledSentence(line, ['TEST_POS_%s' % str(i)])
	labeled_test_pos.append(sentence)
        i += 1

    i = 0
    for line in test_neg:
	sentence = LabeledSentence(line, ['TEST_NEG_%s' % str(i)])
	labeled_test_neg.append(sentence)
        i += 1

    # Initialize model
    model = Doc2Vec(min_count=1, window=10, size=100, sample=1e-4, negative=5, workers=4)
    sentences = labeled_train_pos + labeled_train_neg + labeled_test_pos + labeled_test_neg
    model.build_vocab(sentences)

    # Train the model
    # This may take a bit to run 
    for i in range(5):
        print "Training iteration %d" % (i)
        random.shuffle(sentences)
        model.train(sentences)

    # Use the docvecs function to extract the feature vectors for the training and test data
    train_pos_vec = []
    train_neg_vec = []
    test_pos_vec = []
    test_neg_vec = []
    
    for i in range(len(train_pos)):
	train_pos_vec.append(model.docvecs['TRAIN_POS_%s' % str(i)])

    for i in range(len(train_neg)):
	train_neg_vec.append(model.docvecs['TRAIN_NEG_%s' % str(i)])

    for i in range(len(test_pos)):
	test_pos_vec.append(model.docvecs['TEST_POS_%s' % str(i)])

    for i in range(len(test_neg)):
	test_neg_vec.append(model.docvecs['TEST_NEG_%s' % str(i)])
    
    # Return the four feature vectors
    return train_pos_vec, train_neg_vec, test_pos_vec, test_neg_vec



def build_models_NLP(train_pos_vec, train_neg_vec):
    """
    Returns a BernoulliNB and LosticRegression Model that are fit to the training data.
    """
    Y = ["pos"]*len(train_pos_vec) + ["neg"]*len(train_neg_vec)

    # Use sklearn's BernoulliNB and LogisticRegression functions to fit two models to the training data.
    # For BernoulliNB, use alpha=1.0 and binarize=None
    # For LogisticRegression, pass no parameters
    train_vec = []
    train_vec.extend(train_pos_vec)
    train_vec.extend(train_neg_vec)
    
    nb_model = BernoulliNB(alpha=1.0, binarize=None, class_prior=None, fit_prior=True)
    nb_model.fit(train_vec, Y)
    
    lr_model = LogisticRegression()
    lr_model.fit(train_vec, Y)
    
    return nb_model, lr_model



def build_models_DOC(train_pos_vec, train_neg_vec):
    """
    Returns a GaussianNB and LosticRegression Model that are fit to the training data.
    """
    Y = ["pos"]*len(train_pos_vec) + ["neg"]*len(train_neg_vec)

    # Use sklearn's GaussianNB and LogisticRegression functions to fit two models to the training data.
    # For LogisticRegression, pass no parameters
    train_vec = []
    train_vec.extend(train_pos_vec)
    train_vec.extend(train_neg_vec)

    nb_model = GaussianNB()
    nb_model.fit(train_vec, Y)

    lr_model = LogisticRegression()
    lr_model.fit(train_vec, Y)
    
    return nb_model, lr_model


def evaluate_model(model, test_pos_vec, test_neg_vec, print_confusion=False):
    """
    Prints the confusion matrix and accuracy of the model.
    """
    # Use the predict function and calculate the true/false positives and true/false negative.

    original = ["pos"]*len(test_pos_vec) + ["neg"]*len(test_neg_vec)

    test_vec = []
    test_vec.extend(test_pos_vec)
    test_vec.extend(test_neg_vec)
    predicted = model.predict(test_vec)

    tp = 0
    fn = 0
    fp = 0
    tn = 0

    for i in range(len(original)):
	if original[i]=='pos' and predicted[i] == 'pos':
		tp += 1
        elif original[i]=='pos' and predicted[i] == 'neg':
		fn += 1
        elif original[i]=='neg' and predicted[i] == 'pos':
		fp += 1
        else:
		tn += 1

    accuracy = float(tp + tn)/float(tp + tn + fp + fn)
    
    if print_confusion:
        print "predicted:\tpos\tneg"
        print "actual:"
        print "pos\t\t%d\t%d" % (tp, fn)
        print "neg\t\t%d\t%d" % (fp, tn)
    print "accuracy: %f" % (accuracy)


if __name__ == "__main__":
    main()
