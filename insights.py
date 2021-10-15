import spacy as sp
import pandas as pd
import numpy as np
#loading the english module
nlp= sp.load('en_core_web_sm')

df = pd.read_csv('review_data.csv')

df.info()

#converting amazon pro reviews into a document
pros_amazon=df[df['company']=='amazon']['pros']
pros_amazon
pros_amazon.to_csv('pandas.txt', header=None, index=None, quoting = None,sep=' ', mode='a')

#loading the document as text
with open('pandas.txt', 'r',encoding='utf8') as file:
    txt = file.read().replace('\n', '')

            
txt = txt[:999999]

#applying spacy pipeline on the document
#does all the work in advance
#creating a doc object which we can use to access the methods
doc= nlp(txt)
doc

# spacy provides a one-stop-shop for tasks commonly used in any NLP project
# including:
# Tokenisation
# Lemmatisation
# Part-of-speech tagging
# Entity recognition
# Dependency parsing
# Sentence recognition
# Word-to-vector transformations
# Many convenience methods for cleaning and normalising text


#Tokenization
#Tokenisation is a foundational step in many NLP tasks. 
# Tokenising text is the process of splitting a piece of text into words, 
# symbols, punctuation, spaces and other elements, thereby creating “tokens”.

token_list = []
for token in doc:
    token_list.append(token.text)
print(token_list[:100])

# We can see from the output that it doesn't remove punctuation (',",!,.,etc)
# It also doesn't split verb and adverb ("was", "n't")
# we need to clean this by removing stop words and punctuation
#Stop words with SpaCy

stop = sp.lang.en.stop_words.STOP_WORDS
print('Number of Stop words:',len(stop))
print('Sample Stop Words:',list(enumerate(stop,10))[:5])


#Python provides a list of punctuation marks
import string
print(string.punctuation)

#But we can also use Spacy's is_punct and is_stop to clean them

token_clean_text = []
for token in doc:
    if token.is_punct == False and token.is_stop == False:
        token_clean_text.append(token.text)

print(token_clean_text[:100])
#we can see that we successfully removed stop words and punctuation.
#Now that we removed this, we can start do more text cleaning with spacy.

#we will do Lemmatization next, which are a form of normalizing text by reducing
#words to their original base version ,for example "Runner, Running, Runs" can
#all be boiled down to "Run"

tokens = []
for token in doc:
    if token.is_punct == False and token.is_stop == False:
        tokens.append(token)

for x in tokens[:100]:
    print(x,x.lemma_)
#We can see that "communicated" is reduced to "communicate", "friends" to "friend", etc.

#Now for Part of Speech Tagging "POS"
#A word’s part of speech defines its 
# function within a sentence.
for x in tokens[:100]:
    print(x,x.pos_)
#we can see that it defines stuff like
#  Adjectives, Nouns, Verbs, Numbers,etc..


#Entity Detection helps identify
#important elements like places, people,
#organizations, dates within a text.
#We can visualize it using the Displacy package from spacy.
sp.displacy.render(doc, style = "ent",jupyter = True)


#Dependency Parsing
#It is used to show how a sentence is structured, 
#which helps us better
#determine the meaning of a sentence.
#It is also included in spacy and
#can be visulaised using displacy package

sample = txt.split('"')[1][:50]
test = nlp(sample)
sp.displacy.render(test,style='dep',jupyter=True)

#Word Vector Representation
#This is a necessary step in order to build a classification model for
#sentiment analysis 

#Each word is represented as a vector of numbers, and those numbers
#communicate the relationship of the word to other words, similar to how
#GPS coordinates work.

#Example vector
amazon_vector = nlp('amazon')
print(amazon_vector.vector.shape)
print(amazon_vector.vector)

#We can this use these vectors for text classification with other libraries.

from wordcloud import WordCloud as wc
len(tokens)
l = " ".join([token.orth_ for token in tokens])

cloud = wc(width=1600,height=800,collocations=False).generate(l)


l = " ".join([token.orth_ for token in tokens])

import matplotlib.pyplot as plt

plt.figure(figsize=[20,10])
plt.imshow(cloud, interpolation='bilinear')
plt.axis("off")


df['company'].value_counts()
cons_ms =df[df['company']=='microsoft']['cons']
cons_ms
cons_ms.to_csv('ms_cons.txt', header=None, index=None, quoting = None,sep=' ', mode='a')

with open('ms_cons.txt', 'r',encoding='utf8') as file:
    txt = file.read().replace('\n', '')

            
txt = txt[:999999]

#applying spacy pipeline on the document
#does all the work in advance
#creating a doc object which we can use to access the methods
doc= nlp(txt)

ms_tokens = []
for token in doc:
    if token.is_punct == False and token.is_stop == False:
        ms_tokens.append(token)


l_ms = " ".join([token.orth_ for token in ms_tokens])


ms_cloud = wc(width=1600,height=800,collocations=False,background_color='white').generate(l_ms)


plt.figure(figsize=[20,10])
plt.imshow(ms_cloud,interpolation='bilinear')
plt.axis("off")

##########################################
df['company'].value_counts()
df.info()
summary_apple =df[df['company']=='apple']['summary']
summary_apple 
summary_apple .to_csv('apple_summary.txt', header=None, index=None, quoting = None,sep=' ', mode='a')

with open('apple_summary.txt', 'r',encoding='utf8') as file:
    txt = file.read().replace('\n', '')

            
txt = txt[:999999]

#applying spacy pipeline on the document
#does all the work in advance
#creating a doc object which we can use to access the methods
doc= nlp(txt)

apple_tokens = []
for token in doc:
    if token.is_punct == False and token.is_stop == False:
        apple_tokens.append(token)


l_apple = " ".join([token.orth_ for token in apple_tokens])


apple_cloud = wc(width=1600,height=800,collocations=False,colormap='autumn',background_color='white').generate(l_apple)


plt.figure(figsize=[20,10])
plt.imshow(apple_cloud,interpolation='bilinear')
plt.axis("off")


###################################################
df['company'].value_counts()
df.info()
google_adv =df[df['company']=='google']['advice.to.mgmt']
google_adv 
google_adv .to_csv('google_adv.txt', header=None, index=None, quoting = None,sep=' ', mode='a')

with open('google_adv.txt', 'r',encoding='utf8') as file:
    txt = file.read().replace('\n', '')

            
txt = txt[:999999]

#applying spacy pipeline on the document
#does all the work in advance
#creating a doc object which we can use to access the methods
doc= nlp(txt)

g_tokens = []
for token in doc:
    if token.is_punct == False and token.is_stop == False:
        g_tokens.append(token)


l_g = " ".join([token.orth_ for token in g_tokens])


g_cloud = wc(width=1600,height=800,collocations=False,colormap='Spectral',background_color='black').generate(l_g)

t = [x for x in l_g.split() if x != 'none']
g_cloud = wc(width=1600,height=800,collocations=False,colormap='Spectral',background_color='black').generate(" ".join(t))


plt.figure(figsize=[20,10])
plt.imshow(g_cloud,interpolation='bilinear')
plt.axis("off")


######################################################################


df.info()
#### dates visualization
df['year'].plot(kind='bar')
import seaborn as sns
df['year'] = df['year'].astype(np.int64)

df['year'].value_counts().plot(kind='bar')
df['year'].fillna(2016,inplace=True)
sns.countplot(df['year'])



#### location Visualization
df['location'].value_counts().head(10)

df['location'].value_counts()[1:10].plot(kind='bar')
#employee.status visualization
df['employee.status'].value_counts()
import matplotlib.pyplot as plt
plt.bar(df['employee.status'].value_counts().index,df['employee.status'].value_counts().values)

#calculated_rating average

df.info()

sns.barplot(x=df.groupby('company').mean()['overall.ratings'].index,y=df.groupby('company').mean()['overall.ratings'])

sns.barplot(df.groupby('company').mean()['overall.ratings'].sort_values().index,
df.groupby('company').mean()['overall.ratings'].sort_values().values
,palette='CMRmap_r')








####################################### fucking prediction model #############
df.info()

df_pros = df['pros']

df_pros['type'] = 1

df_pros


df_pros.drop('type',inplace=True)

dfp = df_pros.copy()
import pandas as pd
import numpy as np

df_pos = pd.DataFrame(dfp)

df_pos.info()
df_pos['type'] = 1

dfn = df['cons'].copy()

df_neg = pd.DataFrame(dfn)
df_neg['type'] = 0

df_pos.rename(columns={'pros':'review'},inplace=True)
df_neg.rename(columns={'cons':'review'},inplace=True)

cdf = df_pos.append(df_neg)

model_df = cdf.copy()

model_df.reset_index(drop=True,inplace=True)

model_df

mdf = model_df.sample(frac=1).reset_index(drop=True)

ndf = mdf.copy()

ndf

import string
from spacy.lang.en.stop_words import STOP_WORDS
import spacy as sp
import pandas as pd
import numpy as np
from spacy.lang.en import English

#loading the English module
nlp= sp.load('en_core_web_sm')
punctuations = string.punctuation
stop_words = STOP_WORDS
#parser = English()

#Creating tokenizer function
def spacy_tokenizer(sentence):
    #Creating the token object.
    mytokens = nlp(sentence)
    #Lemmatizing each token and converting each token into lowercase
    mytokens = [ word.lemma_.lower().strip() if word.lemma_ != "-PRON-" else word.lower_ for word in mytokens ]
    #Removing stop words
    mytokens = [ word for word in mytokens if word not in stop_words and word not in punctuations ]
    #Returning preprocessed list of tokens
    return mytokens

#Importing thenecessary libraries 
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.base import TransformerMixin
from sklearn.pipeline import Pipeline

#Creating a Custom transformer using SpaCy
class predictors(TransformerMixin):
    def transform(self, X, **transform_params):
        # Cleaning Text
        return [clean_text(text) for text in X]

    def fit(self, X, y=None, **fit_params):
        return self

    def get_params(self, deep=True):
        return {}

#Creating a Basic function to clean the text
def clean_text(text):
    # Removing whitespaces and converting text into lowercase
    return text.strip().lower()

#CountVectorizer is a great tool provided by the scikit-learn library in Python. 
#It is used to transform a given text into a vector on the basis of the frequency 
#(count) of each word that occurs in the entire text.
#BOW = Bag of Words Matrix, which is what the Countvectorizer generates.
#ngrams is to parse word by word (Unigrams)
bow_vector = CountVectorizer(tokenizer = spacy_tokenizer, ngram_range=(1,1))

#TF-IDF (Term Frequency-Inverse Document Frequency)
#it’s a way of representing how important a particular term is in the context of a 
#given document, based on how many times the term appears and how many other documents 
#that same term appears in. The higher the TF-IDF, the more important that term is.
tfidf_vector = TfidfVectorizer(tokenizer = spacy_tokenizer)

#Splitting the data into Train/Test datasets
from sklearn.model_selection import train_test_split
X = ndf['review'] #The textual reviews
ylabels = ndf['type'] # the labels we want to test against (positive 1 / negative 0)

#splitting 30% of the data into a test dataset to test the accuracy of our model
X_train, X_test, y_train, y_test = train_test_split(X, ylabels, test_size=0.3)

#Using logistic regression
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()

# Create pipeline using Bag of Words
pipe = Pipeline([("cleaner", predictors()),
                 ('vectorizer', bow_vector),
                 ('classifier', classifier)])

#Model generation and training
pipe.fit(X_train,y_train)


#Now to testing the model we generated.
from sklearn import metrics
# Predicting with a test dataset
predicted = pipe.predict(X_test)

# Model Accuracy
print("Logistic Regression Accuracy:",metrics.accuracy_score(y_test, predicted))
print("Logistic Regression Precision:",metrics.precision_score(y_test, predicted))
print("Logistic Regression Recall:",metrics.recall_score(y_test, predicted))



###################
ndf['length'] = ndf['review'].apply(len)

ndf['length'].value_counts()

ndf = ndf[ndf['length'] > 1]

ndf
#########################




plt.style.use('ggplot')
plt.bar(df['employee.status'].value_counts().index,df['employee.status'].value_counts().values)

sns.barplot(df.groupby('company').mean()['overall.ratings'].sort_values().index,
df.groupby('company').mean()['overall.ratings'].sort_values().values
,palette='CMRmap_r')
plt.yticks(range(6))
plt.title('Average Rating by Company')

df.info()