Contents of this folder:
========================


- glove.pkl: Pre-trained GloVe word embeddings, from http://nlp.stanford.edu/data/glove.6B.zip
- GoogleNews-vectors-negative300.bin: Pre-trained word2vec word embeddings, from https://code.google.com/archive/p/word2vec/
- news.pkl: Sentiment Vectors from SocialSent r/news Lexicon, from https://github.com/williamleif/socialsent
- opinion_miner.pkl: Sentiment Vectors from Opinion Mining Lexicon, from http://www.cs.uic.edu/~liub/FBS/opinion-lexicon-English.rar
- README.md: This file.
- sentiwordnet.pkl: Sentiment Vectors from Opinion Mining Lexicon, from https://sentiwordnet.isti.cnr.it/
- svm.pkl: Trained SVM Model.
	- UCI dataset: https://archive.ics.uci.edu/ml/datasets/Sentiment+Labelled+Sentences

	Best parameters set found on development set:

	SVC(C=8.0, cache_size=200, class_weight=None, coef0=0.0,
	  decision_function_shape='ovr', degree=3, gamma='auto', kernel='linear',
	  max_iter=-1, probability=False, random_state=None, shrinking=True,
	  tol=0.001, verbose=False)
	Confusion Matrix
	[[218  69]
	 [ 53 216]]
	Accuracy: 0.780576