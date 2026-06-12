#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


df= pd.read_csv(r"C:/Users/feggi/Downloads/netflix_titles.csv/netflix_titles.csv",encoding='latin =1')


# TASK1: DATASET UNDERSTANDING

# In[3]:


df.head()


# IDENTIFICATION

# In[5]:


#NUMBER OF ROWS AND COLUMNS
rows,column = df.shape
print (f'Rows {rows},Columns {column}')


# In[10]:


#identify the datatypes
df.dtypes


# In[11]:


#Numerical Features
df.describe()


# In[12]:


#including all attributes
df.describe(include='all')


# In[13]:


#checking for primary keys
for col in df.columns:
    if df[col].is_unique:
        print(f"Possible Unique Identifier: {col}")
    else:
        print('null')


# WHAT THE DATASET CONTAINS:
# This dataset was gotten from kaggle.com,it is a CSV file containing data from movies and series released by netflix. It contains a total of 8807 rows and 12 columns.

# TASK 2: CLEANING DATASET

# MISSING VALUES

# In[17]:


#IDENTIFY COLUMNS WITH MISSING VALUES
missing_values= df.columns[df.isnull().any()].tolist()
print(f'The columns with missing values:', missing_values)


# In[19]:


#IDENTIFY THE NUMBER OF MISSING ROWS IN COLUMN
df.isnull().sum()


# DEALING WITH MISSING VALUES

# In[20]:


#FILLING IN THE BLANK CELLS WITH MISSING
# Filling missing values with a placeholder string
df['director'] = df['director'].fillna('missing')
print(df)


# In[21]:


#DOING SAME FOR CAST
# Filling missing values with a placeholder string
df['cast'] = df['cast'].fillna('Missing')
print(df)


# In[22]:


#USING THE MOST OCCURING FOR COUNTRY
mode_value = df['country'].mode()[0]
df['country'] = df['country'].fillna(mode_value)


# In[23]:


print(df)


# NOTE: Filling the dataframe in date_added with the oldest date in the dataset, but first changing the data type to datetime to be on a safe side

# In[24]:


# Convert to standard datetime object
df['date_added'] = pd.to_datetime(df['date_added'].str.strip())

# Fill missing dates with the oldest date in the dataset
df['date_added'] = df['date_added'].fillna(df['date_added'].min())


# In[26]:


#Justing filling the blanks in rating with 'unrated'
df['rating']=df['rating'].fillna('Unrated')


# FILLING THE MISSING CELLS IN COLUMN DURATION
# 1ST; Isolate the numbers
# 2ND; Looking for median
# 3RD; Fill missing cells with median

# In[27]:


#Isolating by cleaning the text 
df['duration_num'] = df['duration'].str.extract('(\d+)').astype(float)

# 2. Get the median of the numeric column
median_duration = df['duration_num'].median()

# 3. Fill missing spots with the median
df['duration_num'] = df['duration_num'].fillna(median_duration)


# In[28]:


#CONFIRMING WORK DONE ON MISSING VALUES
df.isnull().sum()


# In[29]:


#Because of the presence of strings and float
#Safely extract numbers
df['duration_num'] = df['duration'].astype(str).str.extract(r'(\d+)').astype(float)
#Filling missing values based on whether it is a Movie or a TV Show
df['duration_num'] = df.groupby('type')['duration_num'].transform(lambda x: x.fillna(x.median()))
print(df)


# In[30]:


df.isnull().sum()


# In[31]:


# Trying to get duration complete
df['duration'] = df['duration'].fillna(df['duration_num'].astype(int).astype(str) + ' min')
df.isnull().sum()


# In[32]:


#DROPPING THE TEMP. COLUMN
df = df.drop(columns=['duration_num'])
df.head()


# Got the duration complete by filling in the missing cells from 'duration num' and deleted the temporary column because there was no need.

# CHECKING FOR DUPLICATES

# In[34]:


Total_duplicates= df.duplicated().sum()
print(f'Here are the total number of duplicates:', Total_duplicates)


# STANDARDIZATION

# Changed date format from object to time already

# In[35]:


df.dtypes


# In[36]:


# Converting categories with limited choices to 'category' type to save memory
df['rating'] = df['rating'].astype('category')


# In[37]:


df.dtypes


# In[38]:


#TEXT FORMATTING
#Changing object/text columns to lowercase to remove hidden outer whitespaces
text_cols = df.select_dtypes(include=['object']).columns

for col in text_cols:
    df[col] = df[col].astype(str).str.strip().str.lower()


# In[39]:


#Correcting Column Names
# Strip whitespaces, lowercase text, and replace internal spaces with underscores
df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
print(df.columns)


# In[40]:


#Checking for Anomalies
df.describe(include='all')


# SUMMARY TABLE

# In[41]:


#Using Percentile so i can have median included
#50% is the median 
# Specifying 0.50 directly forces the median into the standard describe output
df.describe(percentiles=[0.25, 0.50, 0.75])


# EXPLORATORY ANALYSIS

# In[43]:


#MOVIES VS TV SHOWS DISTRIBUTION

# Counting raw volumes
type_counts = df['type'].value_counts()

# Calculating exact percentages
type_percentages = df['type'].value_counts(normalize=True) * 100

# Combine into a single clean summary table
type_distribution = pd.DataFrame({
    'Total Count': type_counts,
    'Percentage (%)': type_percentages.round(2)
})
print("Movies vs TV Shows Distribution")
print(type_distribution)


# In[44]:


#CONTENT ADDED BY YEAR
# ExtractING the year from the standardized datetime column
df['year_added'] = df['date_added'].dt.year

# Counting per year and sort chronologically
content_by_year = df['year_added'].value_counts().sort_index()

print("Content Added by Year")
print(content_by_year)


# In[46]:


#TOP CONTENT PRODUCING COUNTRIES
#To get indivial countries
# Split comma-separated countries, explode into individual rows, and strip spaces
country_counts = (df['country']
                  .dropna()
                  .str.split(',')
                  .explode()
                  .str.strip()
                  .value_counts())

print("Top Content-Producing Countries")
print(country_counts.head())


# In[47]:


#Caluclating Top ratinng
rating_distribution= df['rating'].value_counts
print('Most Common Rating')
print(rating_distribution)


# In[50]:


# Explode clustered strings into an individual frequency i
genre_counts = (df['listed_in']
                .dropna()
                .str.split(',')
                .explode()
                .str.strip()
                .value_counts())

print("Top Most Common Genres")
print(genre_counts.head())


# DATA VISUALIZATION

# In[55]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Setting Theme
sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.size': 12, 'axes.labelsize': 14, 'axes.titlesize': 16})

# PROCESSING THE DATA FOR CLEAN VISUALS

# Type Distribution Data
type_counts = df['type'].value_counts()

# Year Added Data 
df['year_added'] = pd.to_datetime(df['date_added']).dt.year
year_counts = df['year_added'].value_counts().sort_index()
year_counts = year_counts[year_counts.index >= 2010] 

# Country Data Arrangement
country_counts = df['country'].dropna().str.split(',').explode().str.strip().value_counts().head(6)

# Rating Data Arrangement
rating_counts = df['rating'].dropna().value_counts().head(10)

# Listed_in Data Arrangement
genre_counts = df['listed_in'].dropna().str.split(',').explode().str.strip().value_counts().head(6)


#GENERATING THE ALL THE PLOTS IN ONE 

# Initialize a large figure space to hold all 5 analyses cleanly without crowding
fig = plt.subplots(figsize=(22, 18))


# Movies vs TV Shows Distribution (Donut Chart)

plt.subplot(3, 2, 1)
colors = ['#b20710', '#221f1f'] # USING NETFLIX INSPIRED COLOURS
plt.pie(type_counts, labels=type_counts.index, autopct='%1.1f%%', 
        startangle=90, colors=colors, wedgeprops=dict(width=0.4, edgecolor='w'))
plt.title("Movies vs TV Shows", weight='bold', pad=15)


# Content Added by Year (Line Chart)

plt.subplot(3, 2, 2)
sns.lineplot(x=year_counts.index, y=year_counts.values, marker="o", color='#b20710', linewidth=2.5)
plt.fill_between(year_counts.index, year_counts.values, color='#b20710', alpha=0.1)
plt.title("Platform Growth Timeline (Content Added)", weight='bold', pad=15)
plt.xlabel("Year Added")
plt.ylabel("Count of Titles")


# Top Content-Producing Countries (Bar chart)

plt.subplot(3, 2, 3)
sns.barplot(x=country_counts.values, y=country_counts.index, palette="Reds_r", hue=country_counts.index, legend=False)
plt.title("Content-Producing Countries", weight='bold', pad=15)
plt.xlabel("Total Titles Produced")
plt.ylabel("Country")


#Most Common Audience Ratings (Histogram)

plt.subplot(3, 2, 4)
sns.barplot(x=rating_counts.index, y=rating_counts.values, color='#221f1f')
plt.title("Most Common Audience Ratings", weight='bold', pad=15)
plt.xlabel("Rating Category")
plt.ylabel("Count of Titles")
plt.xticks(rotation=45)


# Most Common Genres/Categories (Bar Chart)

plt.subplot(3, 2, (5, 6)) # Merge the bottom two quadrants into one wide spanning graph
sns.barplot(x=genre_counts.values, y=genre_counts.index, palette="dark:red_r", hue=genre_counts.index, legend=False)
plt.title("Top 6 Most Common Genres ", weight='bold', pad=15)
plt.xlabel("Total Occurrences across Titles")
plt.ylabel("Genre Group")

# Adjust spaces across elements tightly to keep layouts pristine
plt.tight_layout(pad=4.0)


# INSIGHTS
# 1. United States produced more than 60% of the shows released
# 2. Netflix platform is more for the adults than children, with TV-MA having more thn 30% of the uploaded shows
# 3. Netflix started it's peak from year 2015 but showed proper increase in 2016.
