import pandas as pd 

#sample data
data = {
    "label": ["spam", "ham", "spam", "ham"],
    "text": [
        "Win money now!!!",
        "Hey how are you?",
        "Free iPhone!!!",
        "Let's meet tomorrow"
    ]
}

df = pd.DataFrame(data)

print(df.head())
'''  
label                 text
0  spam     Win money now!!!
1   ham     Hey how are you?
2  spam       Free iPhone!!!
3   ham  Let's meet tomorrow
'''

# text to number 
from sklearn.feature_extraction.text import CountVectorizer # it turns text into num by counting words
#it will automatically make all lowercase, remove punctuation, collect all unique words and give them index, convert each sentence into a vector 
#but limits: ignore grammar, word order, ignore meaning
vectorizer = CountVectorizer() # make the object, CountVectorizer() is a class
X = vectorizer.fit_transform(df["text"]) # fit() = read the data / transform() = text to num
# X = matrix

print(X.toarray()) # originally it is in sparse matrix so, .toarray so we can see it
'''
[[0 0 0 0 0 0 0 1 1 0 1 0]
 [1 0 1 1 0 0 0 0 0 0 0 1]
 [0 1 0 0 1 0 0 0 0 0 0 0]
 [0 0 0 0 0 1 1 0 0 1 0 0]]
 '''

#make model
from sklearn.naive_bayes import MultinomialNB
# Naive bayes = probability-based model
# Multinomial = counts words to decides which category is more likely P(spam|words) "given this words how likely is spam"
model = MultinomialNB()
model.fit(X, df["label"]) # X = prob , label = ansewr. now fit the model to this  

#predict 
test = ["Free money!!!"]
test_vec = vectorizer.transform(test)

prediction = model.predict(test_vec)

print(prediction)
