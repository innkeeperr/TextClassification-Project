import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
from sklearn import ensemble
# from keras import layers, models, optimizers
import numpy as np
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords
from textblob import TextBlob
from textblob import Word

def avg_word(sentence):
  words = sentence.split()
  return (sum(len(word) for word in words)/len(words))

display = pd.options.display
display.width = None
display.max_columns = 1000
display.max_rows = 1000
display.max_colwidth = 199

df = pd.read_csv('Reviews.csv', nrows=1000, sep=',')

trainDF = pd.DataFrame()
trainDF["Rating"] = df["Score"]
trainDF["Opinion"] = df["Text"]

# print(trainDF)
# print(trainDF.describe())

# # Utworzenie nowego DataFrame
# train = trainDF.copy()
#
# # wyliczenie ilości słów w opinii
# train['word_count'] = train['Opinion'].apply(lambda x: len(str(x).split(" ")))
# # train[['Opinion', 'word_count']].head()
#
# # Wyliczenie ilości znaków w opinii
# train['char_count'] = train['Opinion'].str.len()
# # train[['Opinion','char_count']].head()
#
# # Średnia długość słowa
# train['avg_word'] = train['Opinion'].apply(lambda x: avg_word(x))
# # train[['Opinion','avg_word']].head()
#
# # Zapisanie stop words w osobnej kolumnie (liczba stopwords)
# stop = stopwords.words('english')
# train['stopwords'] = train['Opinion'].apply(lambda x: len([x for x in x.split() if x in stop]))
# # train[['Opinion', 'stopwords']].head()
#
# # Ile liczb w tekście (do usuniecia)
# train['numerics'] = train['Opinion'].apply(lambda x: len([x for x in x.split() if x.isdigit()]))
# # train[['Opinion', 'numerics']].head()
# print(train)


# OCZYSZCZANIE TEKSTU
# Zmiana liter na małe
trainDF['Opinion'] = trainDF['Opinion'].apply(lambda x: " ".join(x.lower() for x in x.split()))

# usuniecie znaków interpunkcyjnych
trainDF['Opinion'] = trainDF['Opinion'].str.replace('[^\w\s]','')

#Usuniecie z opinii "stop words"
stop = stopwords.words('english')
trainDF['Opinion'] = trainDF['Opinion'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))

# Ktore slowa powtarzaja sie najczesciej + usuniecie ich
freq = pd.Series(' '.join(trainDF['Opinion']).split()).value_counts()[:10]
# print(freq)
freq = list(freq.index)
trainDF['Opinion'] = trainDF['Opinion'].apply(lambda x: " ".join(x for x in x.split() if x not in freq))

# Ktore slowa powtarzaja sie najrzadziej + usuniecie ich
freqMin = pd.Series(' '.join(trainDF['Opinion']).split()).value_counts()[-10:]
# print(freqMin)
freqMin = list(freqMin.index)
trainDF['Opinion'] = trainDF['Opinion'].apply(lambda x: " ".join(x for x in x.split() if x not in freqMin))

# Poprawienie słów, ktore zostały napisane z błędem (funkcja correct() wykonuje się bardzo długo)
# trainDF['Opinion'].apply(lambda x: str(TextBlob(x).correct()))

# Lematyzacja tekstu (jedynie trzon słów), lematyzacja > stemming
trainDF['Opinion'] = trainDF['Opinion'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))

# print(trainDF)
####################################

#split dataset and encode labels
# split the dataset into training and validation datasets
train_x, valid_x, train_y, valid_y = model_selection.train_test_split(trainDF['Opinion'], trainDF['Score'])

# label encode the target variable
encoder = preprocessing.LabelEncoder()
train_y = encoder.fit_transform(train_y)
valid_y = encoder.fit_transform(valid_y)

print(valid_y)


