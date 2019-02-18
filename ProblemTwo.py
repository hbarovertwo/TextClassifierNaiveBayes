import csv
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split

# Preparing the data to train the naive bayes model. Using ItemDescription column to predict PreventiveFlag.
file = csv.DictReader(open('p2_data.csv'))
df = pd.DataFrame(file)
t1 = df['ItemDescription'][0:10000].tolist()
t2 = df['PreventiveFlag'][0:10000].tolist()

# 70/30 split training data set and test data set
xtrain, xtest, ytrain, ytest = train_test_split(t1, t2, test_size=0.3)

# classifier will be a pipeline applying multinomial NB
classifier = make_pipeline(TfidfVectorizer(), MultinomialNB())

# Fitting the model with the training data
classifier.fit(xtrain, ytrain)

# Testing the accuracy of the data with the test data sets.
yscore = classifier.predict(xtest)
num_right = 0
for i in range(len(yscore)):
    if yscore[i] == ytest[i]:
        num_right += 1
accuracy = num_right/float(len(yscore)) * 100
print('Accuracy is ' + str(accuracy) + '%')

# load rest of data, where PreventiveFlag is currently null into classifier
t3 = df['ItemDescription'][10001:].tolist()
pred_probs = []
for i in t3:
    pred_probs.append(classifier.predict_proba([i]).tolist())

# format the output to show probability of predicting 0 or 1
probabilities = []
for i in range(len(pred_probs)):
    probabilities.append('For row ' + str(10001+i) +',P(0) =' +
                         str(pred_probs[i][0][0]*100) + '%, P(1) = ' +
                         str(pred_probs[i][0][1]*100) + '%')

'''
The output asked for is the list probabilities.
It contains the prediction probability for guessing 0 or 1, for each row,
based on the info in the corresponding ItemDescription column.
'''

