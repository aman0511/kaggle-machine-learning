from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from scipy.sparse import hstack, csr_matrix
from nltk.tokenize import wordpunct_tokenize
from nltk.stem.porter import PorterStemmer
from gensim.parsing.preprocessing import preprocess_string
from scipy.stats.stats import pearsonr
import pandas as pd
import numpy as np


df_train = pd.read_csv('toxicity-train.csv')
#df_train = pd.read_csv('toxicity-train.csv', nrows=10000)
df_test = pd.read_csv('toxicity-test.csv')

df_test.fillna('', inplace=True)

print("Train shape = (%d, %d)" % df_train.shape)
print("Test shape = (%d, %d)" % df_test.shape)

X_train = df_train['comment_text']
X_test = df_test['comment_text']

def longest_word_len(comment):
    return max(len(x) for x in comment.split()) if comment and len(comment) else 0

def longest_word_len_under_6(comment):
    lwl = max(len(x) for x in comment.split()) if comment and len(comment) else 0
    return lwl < 6

def count_swear_words(comment):
    top_swear_words = ['shit', 'fuck', 'damn', 'bitch', 'crap', 'piss', 'dick', 'darn', 'cock', 'pussy', 'asshole',
                       'fag', 'bastard', 'slut', 'douche', 'cunt', 'gay', 'bullshit', 'nazi', 'stupid', 'fool',
                       'suck', 'idiot', 'prick', 'jerk', 'moron']
    comment = comment.lower()
    return sum(comment.count(w) for w in top_swear_words)

def count_threats(comment):
    threats = ['kill', 'rape', 'kick']
    comment = comment.lower()
    return sum(comment.count(w) for w in threats)

def count_happy_smilies(comment):
    smilies = [':-)', ';-)', ':)', ';)']
    return sum(comment.count(w) for w in smilies)

def count_bad_smilies(comment):
    smilies = [':-(', ':(']
    return sum(comment.count(w) for w in smilies)

def count_exclamation_marks(comment):
    return comment.count('!')

def count_caps(comment):
    return sum(1 for c in comment if c.isupper())

def caps_vs_total_length(comment):
    return float(count_caps(comment))/float(len(comment)+1)

def count_lines(comment):
    return comment.count('\n')

def count_punctuation(comment):
    return sum(comment.count(w) for w in '.,;:')

def count_words(comment):
    return len(comment.split())

def count_unique_words(comment):
    return len(set(w for w in comment.split()))

def unique_words_vs_length(comment):
    words = [w for w in comment.split()]
    return float(len(set(words))) / float(len(words) + 1)

feature_functions = [longest_word_len, longest_word_len_under_6, count_swear_words, count_happy_smilies, len,
                     count_bad_smilies, count_exclamation_marks, count_caps, caps_vs_total_length, count_lines,
                     count_punctuation, count_threats, count_words, count_unique_words, unique_words_vs_length]

for ff in feature_functions:
    feature_values = df_train['comment_text'].map(ff).values
    print("Correlation {:<32s} with toxic => {:>5.2f} severe_toxic => {:>5.2f} obscene => {:>5.2f} threat => {:>5.2f} insult => {:>5.2f} identity_hate => {:>5.2f}".format(
        ff.__name__,
        pearsonr(df_train['toxic'].values,feature_values)[0],
        pearsonr(df_train['severe_toxic'].values, feature_values)[0],
        pearsonr(df_train['obscene'].values, feature_values)[0],
        pearsonr(df_train['threat'].values, feature_values)[0],
        pearsonr(df_train['insult'].values, feature_values)[0],
        pearsonr(df_train['identity_hate'].values, feature_values)[0]
    ))
exit()


#stemmer = PorterStemmer()
#my_tokenizer = lambda sentence: [stemmer.stem(t) for t in wordpunct_tokenize(sentence.lower())]
#my_tokenizer = lambda sentence: [stemmer.stem(t) for t in sentence.lower().split()]
my_tokenizer = lambda s: preprocess_string(s)

char_trigrams = TfidfVectorizer(min_df=10, max_df=0.75, strip_accents='ascii', analyzer='char', ngram_range=(3, 3), sublinear_tf=True)
char_trigrams.fit(list(X_train.values) + list(X_test.values))

# TODO min_df=3 ?!? (Helps score but it seems wrong)
#word_vect = TfidfVectorizer(min_df=50, max_df=0.75, strip_accents='unicode', analyzer='word', token_pattern=r'\w{1,}', sublinear_tf=True, stop_words='english', ngram_range=(2, 2))
word_ngrams = TfidfVectorizer(min_df=3, max_df=0.75, strip_accents='unicode', analyzer='word', tokenizer=my_tokenizer, sublinear_tf=True, stop_words='english', ngram_range=(1, 2))
word_ngrams.fit(list(X_train.values) + list(X_test.values))

#word_unigrams = TfidfVectorizer(min_df=50, max_df=0.75, strip_accents='unicode', analyzer='word', tokenizer=my_tokenizer, sublinear_tf=True, stop_words='english')
#word_unigrams = TfidfVectorizer(min_df=50, max_df=0.75, strip_accents='unicode', analyzer='word', token_pattern=r'\w{1,}', sublinear_tf=True, stop_words='english')
#word_unigrams.fit(list(X_train.values) + list(X_test.values))

# TODO Can we do something with PCA here ?
#pca = PCA()

def calc_feature_sparse(data, feature_function):
    return csr_matrix(np.reshape(data.map(feature_function).values, (data.shape[0], 1)))

X_train = hstack((
    char_trigrams.transform(X_train.values),
    #word_unigrams.transform(X_train.values),
    word_ngrams.transform(X_train.values),
    calc_feature_sparse(X_train, caps_vs_total_length),
))

X_test = hstack((
    char_trigrams.transform(X_test.values),
    #word_unigrams.transform(X_test.values),
    word_ngrams.transform(X_test.values),
    calc_feature_sparse(X_test, caps_vs_total_length),
))

print("Train shape = (%d, %d)" % X_train.shape)
print("Test shape = (%d, %d)" % X_test.shape)

# scores = cross_val_score(LogisticRegression(C=4.0), X_train, df_train['toxic'].values, cv=5, scoring='neg_log_loss')
# print("Score: {:.4f} (+/- {:.4f})".format(scores.mean(), scores.std()))
# exit()

y_columns = ('toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate')

df_out = df_test[['id']]

for y_col in y_columns:
    #model = LogisticRegression(C=7.0, class_weight={0: 1, 1: 1.75})
    model = LogisticRegression(C=4.0)
    model.fit(X_train, df_train[y_col].values)
    df_out[y_col] = model.predict_proba(X_test)[:,1]

df_out.to_csv('toxicity-submission-wnc.csv', index=False)
