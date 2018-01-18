from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import FunctionTransformer
from sklearn.model_selection import cross_val_score, ShuffleSplit
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC
from nltk.tokenize import wordpunct_tokenize
from nltk.stem.porter import PorterStemmer
from gensim.parsing.preprocessing import preprocess_string
import pandas as pd
from time import time

# http://scikit-learn.org/stable/auto_examples/text/document_classification_20newsgroups.html
# http://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html

df_train = pd.read_csv('toxicity-train.csv')

#X = df_train[['id', 'comment_text']]
X = df_train['comment_text']
y = df_train['toxic']

# print(y.describe())
# exit()

# print("X.shape => (%d)" % X.shape)
# print("y.shape => (%d)" % y.shape)

stemmer = PorterStemmer()
my_wordpunct_tokenizer = lambda sentence: [stemmer.stem(t) for t in wordpunct_tokenize(sentence)]
my_stemmed_tokenizer = lambda sentence: [stemmer.stem(t) for t in sentence.split()]

vect50 = TfidfVectorizer(min_df=50, max_df=0.2, stop_words='english')
vect100 = TfidfVectorizer(min_df=100, max_df=0.2, stop_words='english')
vect250 = TfidfVectorizer(min_df=250, max_df=0.2, stop_words='english')

vect_best = TfidfVectorizer(min_df=3, max_df=0.9, max_features=None, strip_accents='unicode',
                            analyzer='word', token_pattern=r'\w{1,}', ngram_range=(1,2), use_idf=1,
                            smooth_idf=1, sublinear_tf=1, stop_words='english')

vect_char = TfidfVectorizer(min_df=10, max_df=0.9, max_features=None, strip_accents='unicode',
                            analyzer='char', ngram_range=(3,3), use_idf=1, smooth_idf=1, sublinear_tf=1)

#vectorizer = TfidfVectorizer(min_df=250, max_df=0.2, stop_words='english', tokenizer=my_wordpunct_tokenizer)
#vectorizer = TfidfVectorizer(min_df=250, max_df=0.2, stop_words='english', tokenizer=my_stemmed_tokenizer)
#vectorizer = TfidfVectorizer(min_df=250, max_df=0.2, stop_words='english', tokenizer=preprocess_string)

#start = time()
#X_trans = vectorizer.fit_transform(X)
#end = time()
#print("TfidfVectorizer took %.1f" % (end-start))

gpipe = make_pipeline(
    FunctionTransformer(lambda x: x.todense(), accept_sparse=True),
    GaussianNB()
)

models = [
    #('GaussianNB()', gpipe),
    #('LinearSVC()', LinearSVC()),
    #('LinearSVC(class_weight={0:1,1:5})', LinearSVC(class_weight={0: 1.0, 1: 5.0})),
    #('LinearSVC(class_weight={0:1,1:10})', LinearSVC(class_weight={0: 1.0, 1: 10.0})),
    #('LinearSVC(class_weight=\'balanced\')', LinearSVC(class_weight='balanced')),
    #('LogisticRegression()', LogisticRegression()),
    #('LogisticRegression(class_weight={0:1,1:5})', LogisticRegression(class_weight={0: 1.0, 1: 5.0})),
    #('LogisticRegression(class_weight={0:1,1:10})', LogisticRegression(class_weight={0: 1.0, 1: 10.0})),
    #('LogisticRegression(class_weight=\'balanced\')', LogisticRegression(class_weight='balanced')),

    #('TfIdfVect(min=50) -> LogisticRegression()', make_pipeline(vect50, LogisticRegression())),
    #('TfIdfVect(min=50) -> LogisticRegression(C=4.0)', make_pipeline(vect50, LogisticRegression(C=4.0))),
    #('TfIdfVect(min=100) -> LogisticRegression()', make_pipeline(vect100, LogisticRegression())),
    #('TfIdfVect(min=250) -> LogisticRegression()', make_pipeline(vect250, LogisticRegression())),
    #('TfIdfVect(best) -> LogisticRegression()', make_pipeline(vect_best, LogisticRegression())),
    ('TfIdfVect(best) -> LogisticRegression(C=4.0)', make_pipeline(vect_best, LogisticRegression(C=4.0))),
    ('TfIdfVect(char_ngram) -> LogisticRegression()', make_pipeline(vect_char, LogisticRegression(C=4.0))),

]

for label, model in models:
    start = time()
    scores = cross_val_score(model, X, y, cv=ShuffleSplit(n_splits=10, test_size=.2))
    end = time()
    print("{:<48} score: {:.4f} (+/- {:.4f}) [took {:.1f}]".format(label, scores.mean(), scores.std(), end-start))
