####################################################################################
####################################################################################
# # # # # # # # # # # # # # # # # EXERCISE 2 # # # # # # # # # # # # # # # # # # #
####################################################################################
####################################################################################

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

# Load the data from File2.csv
data = pd.read_csv('File2.csv')

# Create a TF-IDF vectorizer to convert text data into numerical features
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(data['OBJECTIVE_NAME'])

TOPICS = ('Water Supply and Drinking Water',
          'Sanitation and Sewage',
          'Environmental Protection',
          'Infrastructure and Services',
          'Economic and Financial Aspects',
          'Social Inclusion and Access',
          'Waste Management',
          'Compliance and Governance',
          'Energy Efficiency'
          )

# Perform K-means clustering to identify main topics
k = 5  # You can adjust the number of clusters as needed
kmeans = KMeans(n_clusters=k, random_state=0)
data['TOPICS'] = kmeans.fit_predict(X)

# Save the results to a CSV file
data.to_csv('File2_Exercise2.csv', index=False)


####################################################################################
####################################################################################
# Explanation of the code:
####################################################################################
####################################################################################

# 1º Step: I start by loading the data from File2.csv, which contains the objectives.
# 2º Step: I use TF-IDF vectorization to convert the text data into numerical features.
# This step helps in representing the text data in a format suitable for clustering.
# 3º Step: I perform K-means clustering with a specified number of clusters (k).
# I also can adjust the number of clusters based on the needs to identify the main topics.
# The TOPICS column contain the cluster labels.
# 4º Step: Finally, I save the results, including the cluster labels, to a CSV file.

# Comment 1:
# It is important to make sure to install the required libraries such as pandas and
# scikit-learn (sklearn) before running the code.
# Comment 2:
# The resulting CSV file (Exercise2_output_candidate.csv) contain the OBJECTIVE_ID,
# OBJECTIVE_NAME, and TOPICS columns, with each objective assigned to a cluster
# label representing a main topic.







