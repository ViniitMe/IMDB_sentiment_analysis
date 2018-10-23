import numpy as np 
import pandas as pd
from bs4 import BeautifulSoup as soup
import re 
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from pandas import Dataframe

path="IMDB_data/labeledTrainData.tsv"
data=pd.read_csv(path,header=0,delimiter="\t",quoting=3)
#header=0 indicates first line of the file contains column names
#quoting=3 tells python to ignore doubled quotes

print(data.shape)
print(data.head())
print(data.info())

#print(data["review"][0])


#Review cleaning function
def review_to_words( raw_review ):
    # Function to convert a raw review to a string of words
    # The input is a single string (a raw movie review), and 
    # the output is a single string (a preprocessed movie review)
    #
    # 1. Remove HTML
    review_text = soup(raw_review).get_text() 
    #
    # 2. Remove non-letters        
    letters_only = re.sub("[^a-zA-Z]", " ", review_text) 
    #
    # 3. Convert to lower case, split into individual words
    words = letters_only.lower().split()                             
    #
    # 4. In Python, searching a set is much faster than searching
    #   a list, so convert the stop words to a set
    stops = set(stopwords.words("english"))                  
    # 
    # 5. Remove stop words
    meaningful_words = [w for w in words if not w in stops]   
    #
    # 6. Join the words back into one string separated by space, 
    # and return the result.
    return( " ".join( meaningful_words ))

#Cleaning all the reviews in the dataset
num_review=len(data['review'])

#empty list
cleaned_review=[]
for i in range(num_review):
	#add to the list
	cleaned_review.append(review_to_words(data['review'][i]))

#view clean review
print(cleaned_review[2])


#Creating bag of words
vectorizer=CountVectorizer(analyzer="word",tokenizer=None,max_features=5000)
train_data_features=vectorizer.fit_transform(cleaned_review)
train_data_features=train_data_features.toarray()
vocub=vectorizer.get_feature_names()
print(train_data_features.shape)

#training the model using random forest
clf=RandomForestClassifier(n_estimators=100)  #100 trees
train=clf.fit(train_data_features,data['sentiment'])


#test data
test_data=pd.read_csv("IMDB_data/testData.tsv",header=0,delimiter="\t",quoting=3)
#test data preparation
num_review=len(test_data['review'])
cleaned_test_review=[]

for i in range(num_review):
	cleaned_test_review.append(review_to_words(test_data['review'][i]))

#note here we just transform not fit as it's test data
test_data_features=vectorizer.transform(cleaned_test_review)
test_data_features=test_data_features.toarray()

result=train.predict(test_data_features)

#copy the result in dataframe
result=pd.Dataframe(data={"id":test_data["id"],"sentiment":result})

result.to_csv("IDMB_model.csv",index=False, quoting=3)


