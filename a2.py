import pandas as pd
import numpy as np
import re
import seaborn as sns
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import mean_squared_error, accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

def valid_isbn(isbn):
    isbn = isbn.strip()

    # Ensures ISBN is either 10 or 13 in length
    if len(isbn) in [10, 13]:
        return isbn
    
    # Checks whether ISBN is a digit and that the final position is either digit or X
    if ((isbn[:-1].isdigit()) and (isbn[-1].isdigit() or isbn[-1].upper() == 'X')):
        return isbn
    
    return None

# Import all necessary CSV
books_df = pd.read_csv('BX-Books.csv')
ratings_df = pd.read_csv('BX-Ratings.csv')
users_df = pd.read_csv('BX-Users.csv')

# Fill in the missing values for book
books_df['Book-Author'].fillna('Unknown', inplace=True)

# Fill the NaN values with 0 in book rating and ensures they are all numeric
ratings_df['Book-Rating'] = ratings_df['Book-Rating'].apply(lambda x: re.sub(r'[^\d+]', '', str(x)))
ratings_df['Book-Rating'] = pd.to_numeric(ratings_df['Book-Rating'], errors='coerce')
ratings_df['Book-Rating'].fillna(0, inplace=True)

# Validate ISBN in both books and ratings csv
books_df['ISBN'] = books_df['ISBN'].astype(str)
books_df['ISBN'] = books_df['ISBN'].apply(lambda x: valid_isbn(x))
ratings_df['ISBN'] = ratings_df['ISBN'].astype(str)
ratings_df['ISBN'] = ratings_df['ISBN'].apply(lambda x: valid_isbn(x))

# Exclude rows with NaN ISBN
books_df['ISBN'] = books_df['ISBN'].dropna()
ratings_df['ISBN'] = ratings_df['ISBN'].dropna()

# Ensures that year are all integers
books_df['Year-Of-Publication'] = books_df['Year-Of-Publication'].apply(lambda x: re.sub(r'[^\d+]', '', str(x)))
books_df['Year-Of-Publication'] = pd.to_numeric(books_df['Year-Of-Publication'], errors='coerce')

# Fills the NaN values in year with the mode
mode_year = books_df['Year-Of-Publication'].mode()[0] 
books_df['Year-Of-Publication'].fillna(mode_year, inplace=True)

# Ensures that age are all integers
users_df['User-Age'] = users_df['User-Age'].apply(lambda x: re.sub(r'[^\d+]', '', str(x)))
users_df['User-Age'] = pd.to_numeric(users_df['User-Age'], errors='coerce')

# Fills the NaN values in age with the value of median frequency
age_counts = users_df['User-Age'].value_counts()
age_median_index = age_counts.sum()/2
age_median = age_counts[(age_counts.cumsum() >= age_median_index)].index[0]
users_df.fillna(age_median, inplace=True)

# Bin the age groups into intervals of 10 
bins = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
labels = ['0-10', '11-20', '21-30', '31-40', '41-50', '51-60', '61-70', '71-80', '81-90', '91-100']
users_df['Age-Label'] = pd.cut(users_df['User-Age'], bins=bins, labels=labels)

# Adjust Data Ranges for Year of Publication
# Assumes the range to be 1900 - 2024
books_df.loc[books_df['Year-Of-Publication'] < 1900, 'Year-Of-Publication'] = 1900
books_df.loc[books_df['Year-Of-Publication'] > 2024, 'Year-Of-Publication'] = 2024

# Adjust rating values to be between 0 and 10
ratings_df.loc[ratings_df['Book-Rating'] < 0, 'Book-Rating'] = 0
ratings_df.loc[ratings_df['Book-Rating'] > 10, 'Book-Rating'] = 10

# Adjust age to be between 0 and 100
users_df.loc[users_df['User-Age'] < 0, 'User-Age'] = 0
users_df.loc[users_df['User-Age'] > 100, 'User-Age'] = 100

# Conduct casefolding and noise removal on the authors
books_df['Book-Author'] = books_df['Book-Author'].str.lower()
books_df['Book-Author'] = books_df['Book-Author'].str.strip() # removes leading and trailing spaces
books_df['Book-Author'] = books_df['Book-Author'].apply(lambda x: re.sub(r'\.(?=[a-zA-Z])', ' ', str(x))) # replaces every dot followed by letter with a space
books_df['Book-Author'] = books_df['Book-Author'].apply(lambda x: re.sub(r'[^A-Za-z\s]', '', str(x))) # removes any punctuation

# Calculate average rating for each author
books_rating = pd.merge(ratings_df, books_df, on='ISBN', how='inner')
author_average_rating = books_rating.groupby('Book-Author')['Book-Rating'].mean().reset_index()
author_average_rating.columns = ['Book-Author', 'AuthorAverage']
merged_df = pd.merge(books_rating, author_average_rating, on='Book-Author', how='inner')

# Calculate trend score based on the year of publication
year_average_rating = books_rating.groupby('Year-Of-Publication')['Book-Rating'].mean().reset_index()
year_average_rating.columns = ['Year-Of-Publication', 'YearAverage']
merged_df = pd.merge(merged_df, year_average_rating, on='Year-Of-Publication', how='inner')
merged_df = pd.merge(merged_df, users_df, on='User-ID', how='inner')
mean_rating = merged_df['Book-Rating'].mean()
merged_df['TopRated'] = (merged_df['Book-Rating'] >= mean_rating).astype(int)

# Plot the distribution of book ratings
sns.countplot(x='Book-Rating', data=ratings_df)
plt.xlabel('Book Rating')
plt.ylabel('Frequency')
plt.title('Distribution of Book Ratings')
plt.show()

# Plot the average rating for each year of publication
sns.lineplot(data=year_average_rating, x='Year-Of-Publication', y='YearAverage')
plt.xlabel('Year of Publication')
plt.ylabel('Average Rating')
plt.title('Average Rating per Year of Publication')
plt.show()

# Plot the average rating for each age group
age_average_rating = merged_df.groupby('Age-Label')['Book-Rating'].mean().reset_index()
sns.barplot(x='Age-Label', y='Book-Rating', data=age_average_rating)
plt.xlabel('Age Group')
plt.ylabel('Average Book Rating')
plt.title('Average Book Rating per Age Group')
plt.show()

# Plot the frequency of ratings given for each age group
age_rating_count = merged_df.groupby('Age-Label')['Book-Rating'].count().reset_index()
plt.figure(figsize=(10, 8))
sns.barplot(x='Age-Label', y='Book-Rating', data=age_rating_count)
plt.xlabel('Age Group')
plt.ylabel('Number of Ratings Given')
plt.title('Number of Ratings Given per Age Group')
plt.show()

# K Nearest Neighbours based on author average rating, year average rating, user age
features = merged_df[['AuthorAverage', 'YearAverage', 'User-Age']]
scaler = StandardScaler()
features_standardized = scaler.fit_transform(features)
predictRating = merged_df['TopRated']

# Uses 20% as the test set and 80% as the training 
X_train, X_test, y_train, y_test = train_test_split(features_standardized, predictRating, test_size=0.2)

# Experiment with different values of k
rms_values = []
k_values = range(1, 11)

for k in k_values:
    knn_model = KNeighborsClassifier(n_neighbors=k, metric='cosine')
    knn_model.fit(X_train, y_train)
    y_pred = knn_model.predict(X_test)
    rms = np.sqrt(mean_squared_error(y_test, y_pred))
    rms_values.append(rms)

# Plot the results
plt.plot(k_values, rms_values, marker='o')
plt.grid(True)
plt.xlabel('K value')
plt.ylabel('RMS')
plt.title('KNN Classifier RMS vs. K Value')
plt.xticks(k_values)
plt.show()

# Based on the graph, we decided to choose k = 9
# Make predictions
knn_model = KNeighborsClassifier(n_neighbors=9, metric='cosine')
knn_model.fit(X_train, y_train)
y_pred = knn_model.predict(X_test)

# Evaluate the model and use performance metrics
rms = np.sqrt(mean_squared_error(y_test, y_pred))
knn_accuracy = accuracy_score(y_test, y_pred)
knn_precision = precision_score(y_test, y_pred)
knn_recall = recall_score(y_test, y_pred)
knn_f1 = f1_score(y_test, y_pred)

print(f"RMS of KNN Classifier: {rms:.2f}") 
print(f"KNN Accuracy: {knn_accuracy:.2f}")
print(f"KNN Precision: {knn_precision:.2f}")
print(f"KNN Recall: {knn_recall:.2f}")
print(f"KNN F1-score: {knn_f1:.2f}")

# Decision Trees based on average author rating, year average rating and age
clf = DecisionTreeClassifier(criterion='entropy')
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)


# Evaluate the accuracy
dt_accuracy = accuracy_score(y_test, y_pred)
dt_precision = precision_score(y_test, y_pred)
dt_recall = recall_score(y_test, y_pred)
dt_f1 = f1_score(y_test, y_pred)

print(f"DT accuracy: {dt_accuracy:.2f}")
print(f"DT Precision: {dt_precision:.2f}")
print(f"DT Recall: {dt_recall:.2f}")
print(f"DT F1-score: {dt_f1:.2f}") 

# Define the metrics and corresponding values for KNN and Decision Tree
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-score']
knn_values = [knn_accuracy, knn_precision, knn_recall, knn_f1]
dt_values = [dt_accuracy, dt_precision, dt_recall, dt_f1]

# Set the width of each bar
bar_width = 0.35

# Set the positions of the bars on the x-axis
# Left for KNN and right for Decision Tree
r1 = np.arange(len(metrics))
r2 = [x + bar_width for x in r1]

# Create the figure and axes
fig, ax = plt.subplots()

# Plot the bars for KNN
ax.bar(r1, knn_values, color='b', width=bar_width, label='KNN')

# Plot the bars for Decision Tree
ax.bar(r2, dt_values, color='r', width=bar_width, label='Decision Tree')

# Add labels, title, and legend
ax.set_xlabel('Metrics')
ax.set_ylabel('Metric Value')
ax.set_title('Comparison of KNN vs. Decision Tree Performance Metrics')
ax.set_xticks(r1 + bar_width / 2)
ax.set_xticklabels(metrics)
ax.legend()
plt.show()