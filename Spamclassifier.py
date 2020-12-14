import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
messages = pd.read_csv('SMSSpamCollection', sep='\t', names = ['label', 'message'])



#Data preprocessing

import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

ps = PorterStemmer()
lemmatizer = WordNetLemmatizer()
corpus = []


for i in range(len(messages)):
    review = re.sub('[^a-zA-Z]', ' ', messages['message'][i])
    review = review.lower()
    review = review.split()
    review = [lemmatizer.lemmatize(word) for word in review if word not in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=5000)
X = cv.fit_transform(corpus).toarray()

y = pd.get_dummies(messages['label'], drop_first=True).values

#train test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

#training model (classic classification problem)

#naive Bayes
from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB()
clf.fit(X_train, y_train)

from sklearn.model_selection import cross_val_score
print(cross_val_score(clf, X_train, y_train).mean())

y_pred = clf.predict(X_test)

from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, y_pred))

#Random Forest
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()
rf.fit(X_train, y_train)

print(cross_val_score(rf, X_train, y_train).mean())

y_pred = rf.predict(X_test)
print(accuracy_score(y_test, y_pred))
