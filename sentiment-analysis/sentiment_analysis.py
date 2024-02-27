import pandas as pd
import numpy as np
import torch
import requests
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from bs4 import BeautifulSoup
import re

# initialise tokenizer and instantiate model, multilingual BERT
tokenizer = AutoTokenizer.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")
model = AutoModelForSequenceClassification.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")

'''
syntax to encode a snetence is as follows:
tokens = tokenizer.encode("Quite happy.", return_tensors = 'pt')
result = model(tokens)
print(result.logits)

return the class(in this case, sentiment rating) with the highest probability
print("The rating is:" , int(torch.argmax(result.logits) + 1))
'''

# replace the URL with the website to be analysed
r = requests.get('https://www.trustpilot.com/review/usacea.org')
soup = BeautifulSoup(r.text, 'html.parser')
# compile regex according to the expression and structure of the website
regex = re.compile('typography_body-l__KUYFJ typography_appearance-default__AAY17 typography_color-black__5LYEn')
results = soup.find_all('p', {'class': regex})
reviews = [result.text for result in results]
# sanity check: uncomment the following line to check if the reviews have parsed correctly
# reviews[0]

df = pd.DataFrame(np.array(reviews), columns=['review'])
# sanity check: uncomment the following line to check what the dataframe looks like
# df.head()
# df.tail()

# based on the syntax for encoding above, this function tokenizes and encodes passed strings
def wrap_tokenizer(review: str)-> int:
    tokens = tokenizer.encode(review, return_tensors = 'pt')
    result = model(tokens)
    return int(torch.argmax(result.logits) + 1)

# use a lambda function to go through each row of df, tokenize it and add results to new column
df['sentiment_rating'] = df['review'].apply(lambda rev: wrap_tokenizer(rev[:-1]))
# sanity check: what does the dataframe look like after the new column has been added
df.head()