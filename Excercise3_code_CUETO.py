####################################################################################
####################################################################################
# # # # # # # # # # # # # # # # # EXERCISE 3 # # # # # # # # # # # # # # # # # # #
####################################################################################
####################################################################################
# Import necessary libraries
import csv
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Define the text classification categories
categories = {
    'A': ['access', 'coverage', 'use', 'reach'],
    'B': ['quality', 'satisfaction', 'continuity'],
    'C': ['management', 'financial', 'performance', 'capacity'],
    'D': []  # 'Other'
}

# Load the data from the CSV file
data = []
with open('data.csv', 'r') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        data.append(row)

# Preprocess the text data
def preprocess_text(text):
    # Tokenization
    tokens = word_tokenize(text.lower())
    # Remove punctuation and stop words
    tokens = [word for word in tokens if word.isalnum() and word not in stopwords.words('english')]
    return ' '.join(tokens)

# Create features (text) and labels (categories)
X = [preprocess_text(row['INDICATOR_NAME']) for row in data]
y = [None] * len(data)

# Classify each indicator based on its keywords
for i, text in enumerate(X):
    category = 'D'  # Default to 'Other'
    for cat, keywords in categories.items():
        if any(keyword in text for keyword in keywords):
            category = cat
            break
    y[i] = category

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorize the text data using a Count Vectorizer
vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

# Train a Multinomial Naive Bayes classifier
classifier = MultinomialNB()
classifier.fit(X_train, y_train)

# Predict categories for the test set
y_pred = classifier.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2%}')

# Classify the entire dataset
X_full = vectorizer.transform(X)
y_full = classifier.predict(X_full)

# Add the predicted categories to the data
for i, row in enumerate(data):
    row['CATEGORY'] = y_full[i]

# Write the results to a new CSV file
with open('output.csv', 'w', newline='') as csvfile:
    fieldnames = ['INDICATOR_ID', 'INDICATOR_NAME', 'CATEGORY']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for row in data:
        writer.writerow(row)

####################################################################################
####################################################################################
# Explanation of the code:
####################################################################################
####################################################################################

#1º Step: the code begins by importing the required Python libraries.
# In this case, it imports several libraries, including the Natural Language Toolkit (NLTK),
# which is a library for working with human language data. Define the text classification categories:
#2º Step: The code defines categories for classifying the text data. In this example,
# there are four categories: A, B, C, and D, with each category associated with specific keywords.
# The goal is to classify the text data into one of these categories based on the presence of keywords.
#3º Step:  The code loads data from a CSV file named 'data.csv.' The data is expected to contain columns
# like 'INDICATOR_NAME' and 'INDICATOR_ID,' which presumably contain indicator names and their corresponding IDs.
#4º Step: In this step, the text data (indicator names) is preprocessed. This preprocessing typically
# involves tokenization (splitting the text into words), converting the text to lowercase, and removing
# punctuation and common stopwords (common words like "the," "and," etc.).
#5º Step: Features (X) are created from the preprocessed text data. These features represent the text
# descriptions of the indicators. Labels (y) are created to represent the categories to which each indicator
# should be classified.
#6º Step: In this step, each indicator is classified into one of the predefined categories (A, B, C, or D)
# based on the presence of keywords. If an indicator contains keywords associated with a particular
# category, it is assigned that category. The default category is 'D' (Other).
#7º Step: The data is split into a training set and a testing set to evaluate the model's performance.
# This is done to assess how well the classification model performs on unseen data.
#8º Step: The text data is transformed into numerical features using a Count Vectorizer.
# This step converts the text data into a numerical format that can be used as input for machine
# learning algorithms.
#9º Step: A Multinomial Naive Bayes classifier is used to train a model for text classification.
# Naive Bayes is a common choice for text classification tasks due to its simplicity and effectiveness.
#10º Step: The trained classifier is used to predict the categories of indicators in the testing set.
#11º Step: The accuracy of the classifier is calculated by comparing the predicted categories with
# the actual categories in the testing set. This gives you an indication of how well the model is performing.
#12º Step: After evaluating the model's performance on the testing set, the model is then used
# to classify the entire dataset, including the training data.
#13º Step: The predicted categories are added to the dataset so that you have a new column ('CATEGORY')
# indicating the category for each indicator.
#14º Step: Finally, the results, including the indicator ID, indicator name, and predicted category,
# are written to a new CSV file named 'output.csv.' This file contains the results of the text classification.


# Comment 1:
# This code essentially demonstrates the process of classifying text data into predefined categories using
# a simple keyword-based approach and machine learning with the Multinomial Naive Bayes classifier.



