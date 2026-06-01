# in this project we will predict whether a news is fake or real 
# here we works on textual data rather than numerical data
# WORKFLOW : News data --> data pre-processing --> spilt data in train and test data -->logistic regression model (binary classification)
# --> trained logistic regression model --> predictions made on new data
# About the dataset : id : unique id for a news article ,title :the title of the news , author : author of the news article
# text : the text of the arcticle ; could be incomplete , label : a label that marks whether the news article is real or fake : 1 : fake ; 0 : real
# importing the dependencies 
import numpy as np # numpy array
import pandas as pd # dataframes
import re  # re stands for regular expressions : useful for searching a text in a document
# nltk : natural language toolkit is a python library for text processing 
# corpus means the body of the text / important content of the text
# stopwords : removes useless words like : the,is,and .
from nltk.corpus import stopwords 
# PorterStemmer : cuts words to their root exm : running-->run , ran --> run
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer # used to convert this text to feature vectors
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import nltk
nltk.download('stopwords')
# print(stopwords.words('english')) # printing the stopwords in english language
# data preprocessing
# load both CSV files
fake_df = pd.read_csv('Fake.csv')
true_df = pd.read_csv('True.csv')
# add a column label : 0 : fake , 1: real
fake_df['label'] = 0
true_df['label'] = 1
# combine both into one dataset 
news_dataset = pd.concat([fake_df,true_df],ignore_index=True)
# shuffle the data so fake/true are mixed
# print(news_dataset.head())
news_dataset = news_dataset.sample(frac=1).reset_index(drop=True)
# print(news_dataset.head())
# print(news_dataset.shape) # prints (rows,columns)
# counting the number of missing values in the dataset
# print(news_dataset.isnull().sum()) 
# skip fillna since we are 0 null values for every column
# otherwise news_dataset = news_dataset.fillna('')
# combining the title and text together and ignoring subject and date
news_dataset['content'] = news_dataset['title'] + ' ' + news_dataset['text']
# print(news_dataset['content'])
# separating the data(content) and labels
X = news_dataset['content']
Y = news_dataset['label']
# print(X)
# print(Y)
# stemming is the process of reducing a word to its root word . exm : actor , actress ,acting --> act
port_stem = PorterStemmer()
stop_words = set(stopwords.words('english'))
def stemming(content) :
    # here we are using regular expression to substiute all the charchacters except a-z and A-Z like numbers ,punctuators with " "
    stemmed_content = re.sub("[^a-zA-Z]"," ",content)
    stemmed_content = stemmed_content.lower() # converting all the characters in lowercase
    stemmed_content = stemmed_content.split()
    # here the for loop is used to call the prt_stem function for all the words except the stopwords
    stemmed_content = [port_stem.stem(word) for word in stemmed_content if not word in stop_words]
    stemmed_content = ' '.join(stemmed_content)
    return stemmed_content
# print("Starting stemming ... this will take 1-2 mins")
news_dataset['content'] = news_dataset['content'].apply(stemming)
# print("Stemming done !")
# print(news_dataset['content'].head())
# separating the data and label
X = news_dataset['content']
Y = news_dataset['label']
# print(X)
# print(Y)
# spliting the dataset to training and test data
X_train , X_test ,Y_train , Y_test = train_test_split(X,Y,test_size=0.2 , stratify=Y, random_state=2)
# print("Train shape : " , X_train.shape)
# print("Test shape : " , X_test.shape)
# converting textual data into numerical data
vectorizer = TfidfVectorizer() # term frequency inverse document frequency , term frequency basically counts the number of times a particular word is repeating in a document
# the repetition tells the model it is a very important word and it assigns a particular numerical value to that word
#  inverse document frequency finds those values which are repeting so many times and it detects that those words are not significant
X_train = vectorizer.fit_transform(X_train)
# transform test using vocab learned from train only
X_test = vectorizer.transform(X_test)
# print("X_train vectorised shape : ",X_train.shape)
#print("X_test vectorised shape : ",X_test.shape)
# training the model : logistic regression
# logistic function / sigmoid function : Y = 1/1+e^-z where Z = w.X+b ; w=weight , b=bias ; it is a S-shaped curve 
# predictive system 
def predict_news(news_text) :
    processed_text = stemming(news_text)
    input_data = vectorizer.transform([processed_text])
    prediction = model.predict(input_data)
    return "The News is Real" if prediction[0] == 1 else "The News is Fake"
if __name__ == "__main__" :
    model = LogisticRegression()
    model.fit(X_train,Y_train)
    train_acc = model.score(X_train,Y_train)
    test_acc = model.score(X_test,Y_test)
    print("Train accuracy : ",train_acc)
    print("Test accuracy : ",test_acc)
    user_input = input("Enter news text :")
    print(predict_news(user_input))