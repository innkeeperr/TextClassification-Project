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


# CONFIGURATION #
display = pd.options.display
display.width = None
display.max_columns = 50
display.max_rows = 50
display.max_colwidth = 199

df = pd.read_csv('Reviews.csv', nrows=10000, sep=',')

# df = pd.DataFrame()
# temp = pd.read_csv('Reviewstest2.csv', iterator=True, chunksize=1000, sep=',')
# df = pd.concat(temp, ignore_index=True)

# CONFIGURATION #

# Klasa służąca do opisu DataFrame'u, tworzy nowe kolumny do DF pokazujące różne parametry tesktu
class TextDescriber:
  def __init__(self, newDataFrame, text, score):
    self.newDataFrame = newDataFrame
    self.text = text
    self.score = score

  # DataFrame printing
  def dfPrint(self):
    print(self.newDataFrame)

  # DataFrame describing
  def dfDescribe(self):
    print(self.newDataFrame.describe())

  # wyliczenie ilości słów w opinii
  def wordCounter (self):
    self.newDataFrame['word_count'] = self.newDataFrame[self.text].apply(lambda x: len(str(x).split(" ")))
    print(self.newDataFrame)

  # Wyliczenie ilości znaków w opinii
  def charCounter (self):
    self.newDataFrame['char_count'] = self.newDataFrame[self.text].str.len()
    print(self.newDataFrame)

  # Średnia długość słowa
  def avgWord(self):
      self.newDataFrame['avg_word'] = self.newDataFrame[self.text].apply(lambda x: sum(len(x) for x in x.split())/len(x.split()))
      print(self.newDataFrame)

  # Zapisanie stop words w osobnej kolumnie (liczba stopwords)
  def stopwordsCounter(self):
    stop = stopwords.words('english')
    self.newDataFrame['stopwords'] = self.newDataFrame[self.text].apply(lambda x: len([x for x in x.split() if x in stop]))

  # Ile liczb w tekście (do usuniecia)
  def numericCounter(self):
    self.newDataFrame['numerics'] = self.newDataFrame[self.text].apply(lambda x: len([x for x in x.split() if x.isdigit()]))



# Główny DataFrame, który zostanie wykorzystany w trenowaniu modelu
trainDF = pd.DataFrame()
trainDF["Rating"] = df["Score"]
trainDF["Opinion"] = df["Text"]

# Utworzenie nowego DataFrame, który posłuży do rozeznania się w tekście
# train = trainDF.copy()
#
# TextAnalyzer = TextDescriber(train, "Opinion", "Rating" )
# TextAnalyzer.wordCounter()
# TextAnalyzer.avgWord()


#################################
# OCZYSZCZANIE TEKSTU

# Klasa pomagająca oczyścić tekst znajdujący się w kolumnie DataFrame
class DataFrameCleaner:
  def __init__(self, mainDF, text, score):
    self.mainDF = mainDF
    self.text = text
    self.score = score

  # DataFrame printing
  def dfPrint(self):
    print(self.mainDF)

  # Zmiana liter na małe
  def dfLowerCase(self):
    self.mainDF[self.text] = self.mainDF[self.text].apply(lambda x: " ".join(x.lower() for x in x.split()))
    return(self.mainDF)

  # Usuniecie znaków interpunkcyjnych
  def dfMarksRemove(self):
    self.mainDF[self.text] = self.mainDF[self.text].str.replace('[^\w\s]', '')
    return self.mainDF

  # Usuniecie z opinii "stop words"
  def dfStopwordsRemove(self):
    stop = stopwords.words('english')
    self.mainDF[self.text] = self.mainDF[self.text].apply(lambda x: " ".join(x for x in x.split() if x not in stop))
    return self.mainDF

  # Usunięcie słów, które powtarzają sie najczęściej
  def dfMostFreqWords(self):
    freq = pd.Series(' '.join(self.mainDF[self.text]).split()).value_counts()[:10]
    freq = list(freq.index)
    self.mainDF[self.text] = self.mainDF[self.text].apply(lambda x: " ".join(x for x in x.split() if x not in freq))
    return self.mainDF

  # Usunięcie słów, które powtarzają sie najrzadziej
  def dfLeastOftenWords(self):
    freqMin = pd.Series(' '.join(self.mainDF[self.text]).split()).value_counts()[-10:]
    freqMin = list(freqMin.index)
    self.mainDF[self.text] = self.mainDF[self.text].apply(lambda x: " ".join(x for x in x.split() if x not in freqMin))
    return self.mainDF

  # Poprawienie słów, ktore zostały napisane z błędem (funkcja correct() wykonuje się bardzo długo)
  def dfCorrectWords(self):
    self.mainDF[self.text].apply(lambda x: str(TextBlob(x).correct()))
    return self.mainDF

  # Lematyzacja tekstu (jedynie trzon słów), lematyzacja > stemming
  def dfLemmatize(self):
    self.mainDF[self.text] = self.mainDF[self.text].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))
    return self.mainDF

# Użycie klasy do oczyszczenia tekstu
mainTrainDF = DataFrameCleaner(trainDF, 'Opinion', 'Rating')
# Pomocniczy obiekt to obliczenia, ile słów zostało usuniętych
TextAnalyzer = TextDescriber(trainDF, "Opinion", "Rating" )
# Przed oczyszczaniem tekstu (print)
# TextAnalyzer.wordCounter()
# Metody
trainDF = mainTrainDF.dfLowerCase()
trainDF = mainTrainDF.dfMarksRemove()
trainDF = mainTrainDF.dfStopwordsRemove()
trainDF = mainTrainDF.dfMostFreqWords()
trainDF = mainTrainDF.dfLeastOftenWords()
trainDF = mainTrainDF.dfLemmatize()
print(trainDF)
# Po oczyszczeniu (print)
# TextAnalyzer.wordCounter()

# END OF TEXT CLEARING #

#####################################

# MODEL TRAINING #
# Podział na zestaw treningowy i walidacyjny
train_x, valid_x, train_y, valid_y = model_selection.train_test_split(trainDF['Opinion'], trainDF['Rating'])
encoder = preprocessing.LabelEncoder()
train_y = encoder.fit_transform(train_y)
valid_y = encoder.fit_transform(valid_y)

# WEKTORYZACJA TEKSTU
# Term frequency & inverse document frequency (word level tf-idf)
tfidf_vect = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=5000)
tfidf_vect.fit(trainDF['Opinion'])
xtrain_tfidf =  tfidf_vect.transform(train_x)
xvalid_tfidf =  tfidf_vect.transform(valid_x)

# N-Gram level tf-idf
tfidf_vect_ngram = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', ngram_range = (2, 3), max_features=5000)
tfidf_vect_ngram.fit(trainDF['Opinion'])
xtrain_tfidf_ngram =  tfidf_vect_ngram.transform(train_x)
xvalid_tfidf_ngram =  tfidf_vect_ngram.transform(valid_x)

#################

def train_model(classifier, feature_vector_train, label, feature_vector_valid, is_neural_net=False):
  # fit the training dataset on the classifier
  classifier.fit(feature_vector_train, label)

  # predict the labels on validation dataset
  predictions = classifier.predict(feature_vector_valid)

  if is_neural_net:
    predictions = predictions.argmax(axis=-1)

  return metrics.accuracy_score(predictions, valid_y)

# Sprawdzeniue różnych klasyfikatorów
# Linear Classifier on Word Level TF IDF Vectors
classifier = linear_model.LogisticRegression()
accuracy = train_model(classifier, xtrain_tfidf, train_y, xvalid_tfidf)
print ("LR, WordLevel TF-IDF: ", accuracy)

# Naive Bayes on Word Level TF IDF Vectors
classifier = naive_bayes.MultinomialNB()
accuracy = train_model(classifier, xtrain_tfidf, train_y, xvalid_tfidf)
print ("NB, WordLevel TF-IDF: ", accuracy)

# SVM on Ngram Level TF IDF Vectors
classifier = svm.SVC()
accuracy = train_model(classifier, xtrain_tfidf_ngram, train_y, xvalid_tfidf_ngram)
print("SVM N-Gram Vectors: ", accuracy)

# RF on Word Level TF IDF Vectors
classifier = ensemble.RandomForestClassifier()
accuracy = train_model(classifier, xtrain_tfidf, train_y, xvalid_tfidf)
print ("RF, WordLevel TF-IDF: ", accuracy)