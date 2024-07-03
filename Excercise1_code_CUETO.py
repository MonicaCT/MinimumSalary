####################################################################################
####################################################################################
# # # # # # # # # # # # # # # # # EXERCISE 1 # # # # # # # # # # # # # # # # # # #
####################################################################################
####################################################################################

import pandas as pd
import re
from googletrans import Translator

# Load the CSV file
df = pd.read_csv('File1.csv')

# Initialize the translator
translator = Translator()

# Create a function to translate text to English
def translate_to_english(text):
    try:
        translated = translator.translate(text, src='auto', dest='en')
        return translated.text
    except Exception as e:
        print(f"Error translating '{text}': {str(e)}")
        return text

# Apply translation to the INDICATOR_NAME_RAW column
df['INDICATOR_NAME_EN'] = df['INDICATOR_NAME_RAW'].apply(translate_to_english)

# Function to remove special characters and clean text
def clean_text(text):
    # Remove special characters, punctuation, and numbers
    text = re.sub(r'[^A-Za-z\s]', '', text)
    # Remove extra spaces and make lowercase
    text = ' '.join(text.split()).lower()
    return text

# Apply text cleaning to the translated column
df['INDICATOR_NAME_CLEAN'] = df['INDICATOR_NAME_EN'].apply(clean_text)

# Save the processed data to a new CSV file
df.to_csv('File1_exercise 1.csv', index=False)

# Print the first few rows of the processed data
print(df.head())

####################################################################################
####################################################################################
# Explanation of the code:
####################################################################################
####################################################################################

# 1º Step: I start by importing the necessary libraries, including pandas for data manipulation,
# the 're' library for regular expressions, and 'googletrans' for translation.
# I needed to install the 'googletrans' library.
# 2º Step: I load the data from File1.csv into a DataFrame using pandas.
# 3º Step: I initialize the Google Translate API for translating the text from different languages to English.
# 4º Step: I define a function named clean_and_translate that takes a text input, translates it to English,
# and removes special characters. Inside the function, I use the Translator to translate the text to English,
# regardless of the input language.
# 5º Step: I remove special characters using a regular expression and substitute them with an empty string.
# This step helps to clean up the text for natural language processing (NLP).
# 6º Step: I apply the clean_and_translate function to each row in the dataset's 'INDICATOR_NAME_RAW' column,
# creating a new 'PROCESSED_INDICATOR_NAME' column to store the cleaned and translated text.
# 7º Step: Finally, I save the processed data to a new CSV file named 'Exercise1_output_candidate.csv'
# with the additional 'PROCESSED_INDICATOR_NAME' column.


# Comment 1:
# The resulting CSV file will contain the original 'INDICATOR_ID' and 'INDICATOR_NAME_RAW' columns,
# along with a new 'PROCESSED_INDICATOR_NAME' column, which contains the processed and translated
# strings ready for NLP tasks. This file serves as the desired output of the code.