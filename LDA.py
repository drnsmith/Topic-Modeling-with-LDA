# -*- coding: utf-8 -*-
"""TM_Assignment_v2.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1ZEnmKEEBVyQOnTxfJ8JZme_uCo5tjmBN
"""

import os
import re
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords', quiet=True)
from nltk.stem import PorterStemmer
from collections import Counter
import string
import pandas as pd
from collections import Counter
import spacy
nltk.download('punkt', quiet=True)
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
nltk.download('averaged_perceptron_tagger', quiet=True)
from nltk.tag import pos_tag
nltk.download('maxent_ne_chunker', quiet=True)
nltk.download('words', quiet=True)
from nltk import CFG
nltk.download('wordnet', quiet=True)
from sklearn.feature_extraction.text import TfidfVectorizer
import gensim
from gensim import corpora, models
from sklearn.decomposition import TruncatedSVD
import wordcloud
import numpy as np
import pandas as pd
from os import path
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from gensim.utils import simple_preprocess
from nltk.corpus import stopwords
import itertools
from itertools import chain
from collections import Counter

# Function to read .xml files in folder_path and return their content in a list
def read_text_files(folder_path):
    blog_content = []
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        if os.path.isfile(file_path) and file_name.endswith('.xml'):
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
                content = file.read()
                blog_content.append(content)

    return blog_content

def remove_non_ascii(text):
    # Remove non-ASCII characters using regex
    return re.sub(r'[^\x00-\x7F]', '', text)

def convert_to_lowercase(text):
    # Convert text to lowercase
    return text.lower()

def remove_noncontext_words(text):

    # Remove newline characters
    text = text.replace('\n', ' ')
    text = text.replace('&nbsp', ' ').replace('nbsp', ' ').replace('&lt', '').replace('&gt', '')
    text = text.replace("[", "").replace("]", "").replace("(", "").replace(")", "").replace("{", "").replace("}", "")
    text = text.replace(",", "").replace("...", "").replace(":", "").replace(";", "").replace("!", ".").replace("?", ".")
    text = re.sub(r"\[(.*?)\]", "", text)  # Remove [+XYZ chars] in content
    text = re.sub(r"\s+", " ", text)  # Remove multiple spaces in content
    text = re.sub(r"\w+…|…", "", text)  # Remove ellipsis (and last word)
    text = re.sub(r"(?<=\w)-(?=\w)", " ", text)  # Replace dash between words
    text_without_urls = re.sub(r"http\S+|\bhttps\S+", "", text) # Remove URLs

    # Tokenize the text into individual words
    words = text.split()

    # Define non-context words to remove
    noncontext_words = ['urllink', 'blog', 'date', 'maio']
    noncontext_words = noncontext_words + ['january', 'february', 'march', 'april', 'may', 'june', 'july', 'august', 'septmber', 'october', 'november', 'december']
    noncontext_words = noncontext_words + ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']
    noncontext_words = noncontext_words + ['mon', 'tue', 'wed', 'thu', 'fri', 'sat', 'sun']
    noncontext_words = noncontext_words + ['nbsp', 'azon', 'acjc', 'alsob']

    # Remove noncontext words from the list of words
    filtered_words = [word for word in words if word not in noncontext_words]

    # Join the filtered words back into a single string
    filtered_text = ' '.join(filtered_words)

    return filtered_text

def remove_stop_words(text):

    # Tokenize the text into individual words
    words = text.split()

    # Stopwords + other words that don't add value to the analysis
    allnonwords = stopwords.words('english') + ['would', 'could', 'said', 'also', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten']
    allnonwords = allnonwords + ['urllink', 'blog', 'date', 'me', 'you', 'she', 'her', 'him', 'they', 'them', 'their', 'your', 'yours', 'our', 'ours']
    allnonwords = allnonwords + ['right', 'look', 'first', 'last', 'never', 'thought', 'next', 'around', 'ever', 'always', 'come', 'to', 'thing', 'things', 'people']
    allnonwords = allnonwords + ['many', 'really', 'thing', 'much', 'stuff', 'there', 'hour', 'point', 'mine', 'yours', 'hers',
                                'his', 'theirs', 'ours', 'issue', 'thing', 'did', 'didnt', 'year', 'maio', 'thing', 'best', 'since', 'month', 'feel']
    allnonwords = allnonwords + ['much', 'something', 'someone', 'even', 'well', 'still', 'little', 'always', 'never', 'ever', 'sure', 'sort']
    allnonwords = allnonwords + ['every', 'anything', 'everything', 'nothing', 'everyone', 'everybody', 'everywhere', 'anyone', 'anybody']
    allnonwords = allnonwords + ['anywhere', 'someone', 'somebody', 'somewhere', 'nowhere', 'thing', 'something', 'nothing', 'everything']
    allnonwords = allnonwords + ['always', 'another', 'though', 'without', 'actually', 'do', 'dont', 'will', 'wont', 'can', 'cant']
    allnonwords = allnonwords + ['get', 'got', 'go', 'going', 'know', 'let', 'like', 'make', 'see', 'want', 'come', 'take', 'think']
    allnonwords = allnonwords + ['back', 'great', 'today', 'year', 'good', 'link', 'night', 'went', 'couple', 'say', 'give', 'need', 'make']
    allnonwords = allnonwords + ['youre', 'youve', 'youll', 'youd', 'hes', 'shes', 'its', 'were', 'theyre', 'thats', 'week', 'made', 'remember',
                                 'might', 'getting', 'better', 'real', 'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaahhhhhhhh', 'news', 'new', 'top',
                                 'u', 'day', 'brureau', 'love', 'u', 'do', 'not', 'well', 'fuck', 'na', 'haha', 'post', 'there',
                                 'anyway', 'ask', 'that', 'mean', 'dunno', 'file', 'miss', 'true', 'point', 'call', 'came', 'look',
                                 'site', 'na', 'talk', 'place', 'need', 'there', 'blog', 'entry', 'originally', 'posted', 'show', 'start',
                                 'okay', 'lots', 'finally', 'yippee', 'comes', 'hello', 'late', 'wish', 'weblog', 'damit', 'dammit',
                                 'currently', 'lala', 'opposite', 'told', 'update', 'updating', 'sometimes', 'maybe',
                                 'easy', 'half', 'different', 'called', 'total', 'took', 'word', 'done', 'stay', 'fine', 'find', 'cannot',
                                 'front', 'back', 'dude', 'feel', 'name', 'time', 'man', 'woman', 'home', 'ching', 'year',
                                 'times', 'yeah', 'sorry', 'whole', 'pretty', 'guess', 'nice', 'tomorrow', 'day']


    # Remove stop words from the list of words
    filtered_words = [word for word in words if word not in allnonwords]

    # Join the filtered words back into a single string
    filtered_text = ' '.join(filtered_words)

    return filtered_text

def remove_short_words(text):
    # Split the text into words
    words = text.split()

    # Remove words with three letters or less
    filtered_words = [word for word in words if len(word) > 3]

    # Join the filtered words back into a text
    filtered_text = ' '.join(filtered_words)

    return filtered_text

def remove_dates(text):
    # Define regex pattern to match months and days of the week
    pattern = r'\b(?:january|february|march|april|may|june|july|august|september|october|november|december|monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b'

    # Remove matched dates using regex substitution
    result = re.sub(pattern, '', text)
    return result

def remove_tags(text):
    # Remove HTML tags using regex
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'<*?>', '', text)
    return text

def remove_numbers(text):
    pattern = r'\d+'  # Regular expression pattern to match numbers
    result = re.sub(pattern, '', text)  # Replace the pattern with empty spaces
    return result

def remove_punctuation_and_newlines(text):
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Remove newline characters
    # text = text.replace('\n', '')
    return text

def lemmatize_text(text):
    # Tokenize the text into individual words
    tokens = word_tokenize(text)

    # Initialize the WordNetLemmatizer
    lemmatizer = WordNetLemmatizer()

    # Lemmatize each token in the text
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]

    # Join the lemmatized tokens back into a single string
    lemmatized_text = " ".join(lemmatized_tokens)

    return lemmatized_text

def stem_text(text):
    # Tokenize the text into words
    words = word_tokenize(text)

    # Initialize Porter stemmer
    stemmer = PorterStemmer()

    # Apply stemming to each word
    stemmed_words = [stemmer.stem(word) for word in words]

    # Join the stemmed words back into a text
    stemmed_text = ' '.join(stemmed_words)

    return stemmed_text

# Functions to expand contractions e.g. don't -> do not
contraction_dict = {"ain't": "is not", "aren't": "are not","can't": "cannot", "'cause": "because", "could've": "could have", "couldn't": "could not", "didn't": "did not",  "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not", "he'd": "he would","he'll": "he will", "he's": "he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is",  "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have","I'm": "I am", "I've": "I have", "i'd": "i would", "i'd've": "i would have", "i'll": "i will",  "i'll've": "i will have","i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would", "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have","it's": "it is", "let's": "let us", "ma'am": "madam", "mayn't": "may not", "might've": "might have","mightn't": "might not","mightn't've": "might not have", "must've": "must have", "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have","o'clock": "of the clock", "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have", "she'd": "she would", "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", "she's": "she is", "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have","so's": "so as", "this's": "this is","that'd": "that would", "that'd've": "that would have", "that's": "that is", "there'd": "there would", "there'd've": "there would have", "there's": "there is", "here's": "here is","they'd": "they would", "they'd've": "they would have", "they'll": "they will", "they'll've": "they will have", "they're": "they are", "they've": "they have", "to've": "to have", "wasn't": "was not", "we'd": "we would", "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", "we're": "we are", "we've": "we have", "weren't": "were not", "what'll": "what will", "what'll've": "what will have", "what're": "what are",  "what's": "what is", "what've": "what have", "when's": "when is", "when've": "when have", "where'd": "where did", "where's": "where is", "where've": "where have", "who'll": "who will", "who'll've": "who will have", "who's": "who is", "who've": "who have", "why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not", "won't've": "will not have", "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all", "y'all'd": "you all would","y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have","you'd": "you would", "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have", "you're": "you are", "you've": "you have"}
def _get_contractions(contraction_dict):
    contraction_re = re.compile('(%s)' % '|'.join(contraction_dict.keys()))
    return contraction_dict, contraction_re

contractions, contractions_re = _get_contractions(contraction_dict)

def replace_contractions(text):
    def replace(match):
        return contractions[match.group(0)]
    return contractions_re.sub(replace, text)

# Remove very rare words (they are probably typos or noise)
def remove_less_frequent_words(text, num_words):
    # Count the occurrences of words in the text
    word_counts = Counter(text.split())

    # Collect the words to be removed
    words_to_remove = set()
    for word, count in word_counts.items():
        if count <= num_words:
            words_to_remove.add(word)

    # Remove the less frequent words from the text
    processed_text = ' '.join(word for word in text.split() if word not in words_to_remove)

    return processed_text


# Go through files and save all filenames in a list
def extract_filenames(directory):
    file_names = []
    for file_name in os.listdir(directory):
        file_path = os.path.join(directory, file_name)
        if os.path.isfile(file_path) and file_name.endswith('.xml'):
            file_names.append(file_name)
    return file_names

# Iterate through the blog files in the directory and save demographic information as dataframe
def extract_demographics(directory, filenames):

    # Initialize demographic information
    blog_ids, genders, ages, educations, starsigns = [], [], [], [], []
    # Iterate through the blog files in the directory
    for filename in filenames:

        # Extract demographics from the filename and save to lists
        file_parts = filename.split('.')
        blog_id = file_parts[0]
        blog_ids.append(blog_id)
        gender = file_parts[1]
        genders.append(gender)
        age = int(file_parts[2])
        ages.append(age)
        education = file_parts[3]
        educations.append(education)
        starsign = file_parts[4]
        starsigns.append(starsign)

    data = {'ID': blog_ids, 'Gender': genders, 'Age': ages, 'Education': educations, 'Starsign': starsigns}
    return pd.DataFrame(data)

# Find nouns and lemmatize them, save to new list clean_content_nouns
def extract_nouns(text):

    lemmatizer = WordNetLemmatizer()
    stemmer = PorterStemmer()

    # Tokenize the text into words
    tokens = word_tokenize(text)

    # Perform part-of-speech tagging
    tagged_words = nltk.pos_tag(tokens)

    # Extract nouns
    nouns = [stemmer.stem(lemmatizer.lemmatize(word)) for word, pos in tagged_words if pos.startswith('NN')]

    return Counter(nouns)

# Find subjects of clauses and lemmatize them, and save to new list clean_content_subjects
def extract_subjects(text):

    lemmatizer = WordNetLemmatizer()
    stemmer = PorterStemmer()

    # Tokenize the text into sentences
    sentences = nltk.sent_tokenize(text)

    subjects = []
    # Iterate over each sentence
    for sentence in sentences:
        # Tokenize the sentence into words
        words = nltk.word_tokenize(sentence)

        # Perform part-of-speech tagging
        tagged_words = nltk.pos_tag(words)

        # Extract subjects of clauses
        for i in range(len(tagged_words)):
            word, pos = tagged_words[i]
            if pos.startswith('V') and i > 0:
                prev_word, prev_pos = tagged_words[i - 1]
                if prev_pos.startswith('N') or prev_pos == 'PRP':
                    subject = stemmer.stem(lemmatizer.lemmatize(prev_word))
                    subjects.append(subject)

    return Counter(subjects)

def find_most_common_word_intext(text):

    lemmatizer = WordNetLemmatizer()
    stemmer = PorterStemmer()

    # Tokenize, lemmatize and stem the text
    words = [stemmer.stem(lemmatizer.lemmatize(word)) for word in word_tokenize(text)]

    # Count the occurrences of each word
    word_counts = Counter(words)
    return word_counts

# Find all clauses in each blog that have the identified topics in them
def find_clauses_with_word(text_list, word_list):
    if len(text_list) != len(word_list):
        raise ValueError("Number of texts and words should be the same.")

    result = []

    for text, word in zip(text_list, word_list):
        clauses = re.findall(r"[^.!?]+", text)  # Split the text into clauses
        matching_clauses = [clause.strip() for clause in clauses if word in clause]  # Find matching clauses
        matching_clauses = " .".join(matching_clauses)  # Join the clauses into a single string
        result.append([matching_clauses])

    return result

# Function to find most important words in list of texts using TF-IDF
def find_most_important_word_TFIDF(texts):
    # Initialize the TF-IDF vectorizer
    vectorizer = TfidfVectorizer()

    # Fit and transform the texts using the vectorizer
    tfidf_matrix = vectorizer.fit_transform(texts)

    # Get the feature names (words)
    feature_names = vectorizer.get_feature_names_out()

    # Find the most important word in each text
    most_important_words = []
    for i in range(len(texts)):

        if len(texts[i]) > 0:

            # Get the TF-IDF scores for the current text
            tfidf_scores = tfidf_matrix[i].toarray()[0]

            # Find the index of the word with the highest TF-IDF score
            max_index = tfidf_scores.argmax()

            # Get the most important word
            most_important_word = feature_names[max_index]

        else:
            most_important_word = ''

        # Append the most important word to the list
        most_important_words.append(most_important_word)

    return most_important_words

# Function to find most common topics in a given field
def find_most_common_topics(df, field, field_filter, topic):

    # Filter the dataframe
    filtered_df = df[(df[field] == field_filter)]

    # Count the occurrences of each topic
    topic_counts = filtered_df[topic].value_counts()

    # Get the top two most common topics
    top_two_topics = topic_counts.nlargest(3)
    print(top_two_topics)
    return top_two_topics

def print_results(df, field, field_filter):
    methods = ['Most Common Word', 'Most Common Noun', 'Most Common Subject', 'Word TFIDF', 'Noun TFIDF', 'Subject TFIDF']
    for method in methods:
        print("Most common topic using " + method + " among " + field + " " + field_filter + " authors:")
        find_most_common_topics(df, field, field_filter, method)


def find_topic_LDA(string_list, num_topics, num_words):

    dictionary = gensim.corpora.Dictionary(string_list)
    #dictionary.filter_extremes(no_below=15, no_above=0.1)

    bow_corpus = [dictionary.doc2bow(doc) for doc in string_list]

    lda_model = gensim.models.LdaMulticore(bow_corpus,
                                        num_topics=num_topics,
                                        id2word = dictionary,
                                        passes = 10,
                                        workers = 5)
    idxs = []
    topics = []
    for idx, topic in lda_model.print_topics(-1):
        idxs.append(idx)
        topics.append(topic)
    return idxs, topics

def find_topics_LDA_group(field, field_value_target, topics_list, blogs_df):
    topics_LDA_sublist = []
    for i, field_value in enumerate(blogs_df[field]):
        if field_value_target == field_value:
            topics_LDA_sublist.append(topics_list[i])
    topics_LDA_sublist_flat = list(itertools.chain(*topics_LDA_sublist))
    return Counter(topics_LDA_sublist_flat).most_common(3)

def print_results_LDA(field, field_value, topics_lda, blogs_df):
    print("Most common topic using LDA among " + field + " " + field_value + " authors:")
    print(find_topics_LDA_group(field, field_value, topics_lda, blogs_df))

def main():

    import os
    import re
    import nltk
    from nltk.corpus import stopwords
    nltk.download('stopwords', quiet=True)
    from nltk.stem import PorterStemmer
    from collections import Counter
    import string
    import pandas as pd
    from collections import Counter
    import spacy
    nltk.download('punkt', quiet=True)
    from nltk.stem import WordNetLemmatizer
    from nltk.tokenize import word_tokenize
    nltk.download('averaged_perceptron_tagger', quiet = True)
    from nltk.tag import pos_tag
    nltk.download('maxent_ne_chunker', quiet=True)
    nltk.download('words', quiet=True)
    from nltk import CFG
    nltk.download('wordnet', quiet=True)
    from sklearn.feature_extraction.text import TfidfVectorizer
    import gensim
    from gensim import corpora, models
    from sklearn.decomposition import TruncatedSVD
    import wordcloud
    import numpy as np
    import pandas as pd
    from os import path
    from PIL import Image
    from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
    import matplotlib.pyplot as plt
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.decomposition import LatentDirichletAllocation
    import gensim
    from gensim.utils import simple_preprocess
    from nltk.corpus import stopwords
    import numpy as np
    import nltk
    import itertools
    from itertools import chain
    from collections import Counter

    #Mounting the drive
    from google.colab import drive
    import os
    import pathlib

    drive.mount('/content/gdrive')
    path = '/content/gdrive/My Drive/Colab Notebooks'
    os.chdir(path)

    #Unzip the files
    #with zipfile.ZipFile(os.path.join(os.getcwd(),'blogs_all.zip'), 'r') as zip_ref:
    #zip_ref.extractall(os.getcwd())

    # Reading and storing the blog data as a list 'blog_content'

    # Specify the path to the 'blogs' directory
    #directory = r"C:\Users\zch161\notebooks\blogs_all"
    directory = '/content/gdrive/My Drive/Colab Notebooks/blogs_all'

    # Get a list of all files in the directory
    file_list = os.listdir(directory)

    # Sort the file list in ascending order
    file_list.sort()

    # Read text files and save their contents in blog_content list
    blog_content = read_text_files(directory)

    # Original content in lowercase
    content = [convert_to_lowercase(text) for text in blog_content]

    # General cleaning
    clean_content = [remove_tags(text) for text in content]
    clean_content = [remove_non_ascii(text) for text in clean_content]
    clean_content = [remove_dates(text) for text in clean_content]
    clean_content = [remove_numbers(text) for text in clean_content]
    clean_content = [replace_contractions(text) for text in clean_content]

    # For use in finding clauses
    clean_content_a = [remove_noncontext_words(text) for text in clean_content]

    # For use in frequency methods and LDA
    clean_content_b = [remove_stop_words(text) for text in clean_content_a]
    clean_content_b = [remove_punctuation_and_newlines(text) for text in clean_content_b]
    clean_content_b = [remove_short_words(text) for text in clean_content_b]
    clean_content_b = [remove_less_frequent_words(text, 3) for text in clean_content_b]

    lemmatized_clean_content_b = [lemmatize_text(text) for text in clean_content_b]
    lemmatized_clean_content_b = [stem_text(text) for text in lemmatized_clean_content_b]
    lemmatized_clean_content_b = [remove_stop_words(text) for text in lemmatized_clean_content_b]
    lemmatized_clean_content_b = [remove_less_frequent_words(text, 3) for text in lemmatized_clean_content_b]

    # Remove non-frequent words to use in TF-IDF
    clean_content_b_TFIDF = lemmatized_clean_content_b.copy()

    ## Finding demographics of blog authors and saving as dataframe 'blogs_df'

    blogs_df = extract_demographics(directory, extract_filenames(directory))

    # Save clean content as csv file
    df_content = pd.DataFrame(data={"Content": lemmatized_clean_content_b, "ID": blogs_df['ID'], "Gender": blogs_df["Gender"],
                                    "Age": blogs_df["Age"], "Education": blogs_df["Education"], "Starsign": blogs_df["Starsign"]})
    df_content.to_csv("./blogs.csv", sep=',',index=False)

    # Finding the number of different groups in the blog author data and printing output

    male_count = blogs_df[blogs_df['Gender'] == 'male'].shape[0]
    female_count = blogs_df[blogs_df['Gender'] == 'female'].shape[0]
    unknown_gender = len(blogs_df) - male_count - female_count

    # Create new column in dataframe that indicates whether the blog author is below 20 years old
    blogs_df['Under 20'] = 'n'
    blogs_df.loc[blogs_df['Age'] < 20, 'Under 20'] = 'y'

    age_below_20 = blogs_df[blogs_df['Age'] < 20].shape[0]
    age_above_20 = blogs_df[blogs_df['Age'] >= 20].shape[0]
    unknown_age = len(blogs_df) - age_below_20 - age_above_20

    students = blogs_df[blogs_df['Education'] == 'student'].shape[0]
    unknown_occupation = len(blogs_df) - students

    print('Number of male, female and unknown gender blog authors: ', str(male_count) + ', ' + str(female_count) + ', ' + str(unknown_gender))
    print('Number of blog authors below 20, above 20 and unknown age: ', str(age_below_20) + ', ' + str(age_above_20) + ', ' + str(unknown_age))
    print('Number of students and unknown occupation blog authors: ', str(students) + ', ' + str(unknown_occupation))
    print('Total number of blogs: ', len(blogs_df))

    ## Create two new lists from corpus, one with nouns only, one with subjects only

    clean_content_nouns = [extract_nouns(text) for text in clean_content_b]
    print('Noun extraction done')
    clean_content_subjects = [extract_subjects(text) for text in clean_content_b]
    print('Subject extraction done')

    ## Finding topics using frequency

    # Find topics of blog posts by finding the n most common WORDs as list most_common_words
    # Add this to the dataframe blogs_df

    # Find the most common word in each blog post and add to blogs_df
    num_words = 5
    most_common_words = [find_most_common_word_intext(text).most_common(num_words) for text in lemmatized_clean_content_b]
    blogs_df['Most Common Word'] = [sublist[0][0] if len(sublist) > 0 else '' for sublist in most_common_words]
    blogs_df['MCW Count'] = [sublist[0][1] if len(sublist) > 0 else 0 for sublist in most_common_words]
    blogs_df['Clauses w/ MCW'] = [clauses for clauses in find_clauses_with_word(clean_content_a, [sublist[0][0] if len(sublist) > 0 else 'dummy_clause' for sublist in most_common_words])]
    print(blogs_df.head())

    # Find topics of blog posts by finding the most common NOUN as list most_common_nouns
    # Add this to the dataframe blogs_df
    most_common_nouns = [extract_nouns(text).most_common(num_words) for text in clean_content_b]
    blogs_df['Most Common Noun'] = [sublist[0][0] if len(sublist) > 0 else '' for sublist in most_common_nouns]
    blogs_df['MCN Count'] = [sublist[0][1] if len(sublist) > 0 else 0 for sublist in most_common_nouns ]
    blogs_df['Clauses w/ MCN'] = [clauses for clauses in find_clauses_with_word(clean_content_a, [sublist[0][0] if len(sublist) > 0 else 'dummy_clause' for sublist in most_common_nouns ])]

    # Find topics of blog posts by finding the most common CLAUSE SUBJECT as list most_common_subjects
    # Add this to the dataframe blogs_df
    most_common_subjects = [extract_subjects(text).most_common(num_words) for text in clean_content_b]
    blogs_df['Most Common Subject'] = [sublist[0][0] if len(sublist) > 0 else '' for sublist in most_common_subjects]
    blogs_df['MCS Count'] = [sublist[0][0] if len(sublist) > 0 else 0 for sublist in most_common_subjects ]
    blogs_df['Clauses w/ MCS'] = [clauses for clauses in find_clauses_with_word(clean_content_a, [sublist[0][0] if len(sublist) > 0 else 'dummy_clause' for sublist in most_common_subjects])]

    print(blogs_df.head())

    ## Finding topics using TF-IDF

    # Find most important words in each blog post using TF-IDF and all WORDS
    most_important_words = find_most_important_word_TFIDF(clean_content_b_TFIDF)
    # Add to blogs_df

    blogs_df['Word TFIDF'] = [word if len(word) > 0 else '' for word in most_important_words]
    blogs_df['Clauses w/ word'] = [clauses for clauses in find_clauses_with_word(clean_content_a, [word if len(word) > 0 else '' for word in most_important_words])]

    clean_content_nouns_string = [" ".join(nouns) for nouns in clean_content_nouns]

    # Find most important words in each blog post using TF-IDF and all NOUNS
    most_important_nouns = find_most_important_word_TFIDF([' '.join(counter.keys()) for counter in clean_content_nouns])
    # Add to blogs_df
    blogs_df['Noun TFIDF'] = [word for word in most_important_nouns]
    blogs_df['Clauses w/ noun'] = [clauses for clauses in find_clauses_with_word(clean_content_a, most_important_nouns)]

    # Find most important words in each blog post using TF-IDF and all SUBJECTS
    most_important_subjects = find_most_important_word_TFIDF([' '.join(counter.keys()) for counter in clean_content_subjects])
    # Add to blogs_df
    blogs_df['Subject TFIDF'] = [word for word in most_important_subjects]
    blogs_df['Clauses w/ subject'] = [clauses for clauses in find_clauses_with_word(clean_content_a, most_important_subjects)]

    print(blogs_df.head(10))

    # Find top topics by all, males, females, under 20, over 20, students

    blogs_df['Under 20'] = 'n'
    blogs_df.loc[blogs_df['Age'] < 20, 'Under 20'] = 'y'
    blogs_df['All'] = 'y'

    # Print the results
    print_results(blogs_df, 'Gender', 'female')
    print_results(blogs_df, 'Gender', 'male')
    print_results(blogs_df, 'Under 20', 'y')
    print_results(blogs_df, 'Under 20', 'n')
    print_results(blogs_df, 'Education', 'Student')
    print_results(blogs_df, 'All', 'y')

    blogs_df.head(500).to_csv('output500.csv', index=False)

    blogs_df.to_csv('output.csv', index=False)

    ## Create Word Cloud with nouns

    content_string = [' '.join(blogs_df["Most Common Noun"].tolist())]

    # lower max_font_size, change the maximum number of word and lighten the background:
    wordcloud = WordCloud(max_font_size=50, max_words=100, background_color="white").generate(content_string[0])
    plt.figure()
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.show()
    wordcloud.to_file("word_cloud.png")

    ## Use LDA to find topics in the blog posts

    import pandas as pd
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.decomposition import LatentDirichletAllocation
    # Step 1: Load and preprocess the data
    # Read the CSV file using pandas
    blogs = pd.read_csv('/content/gdrive/My Drive/Colab Notebooks/blogs.csv')

    # Remove rows with missing values in the 'Content' column
    blogs = blogs.dropna(subset=['Content'])

    # Step 2: Vectorize the text data
    cv = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
    dtm = cv.fit_transform(blogs['Content'])

    # Fit LDA model
    LDA = LatentDirichletAllocation(n_components=7, random_state=42)
    LDA.fit(dtm)

    # Assign topic labels to each document
    topic_results = LDA.transform(dtm)
    blogs['Topic'] = topic_results.argmax(axis=1)

    # Define demographic groups
    male = blogs[blogs['Gender'] == 'male']
    female = blogs[blogs['Gender'] == 'female']
    over_20 = blogs[blogs['Age'] > 20]
    under_20 = blogs[blogs['Age'] <= 20]
    students = blogs[blogs['Education'] == 'Student']

    # Calculate the topic distribution for each demographic
    male_topic_dist = male['Topic'].value_counts(normalize=True)
    female_topic_dist = female['Topic'].value_counts(normalize=True)
    over_20_topic_dist = over_20['Topic'].value_counts(normalize=True)
    under_20_topic_dist = under_20['Topic'].value_counts(normalize=True)
    students_topic_dist = students['Topic'].value_counts(normalize=True)
    everyone_topic_dist = blogs['Topic'].value_counts(normalize=True)

    # Find the two most popular topics for each demographic
    male_popular_topics = male_topic_dist.nlargest(2).index.tolist()
    female_popular_topics = female_topic_dist.nlargest(2).index.tolist()
    over_20_popular_topics = over_20_topic_dist.nlargest(2).index.tolist()
    under_20_popular_topics = under_20_topic_dist.nlargest(2).index.tolist()
    students_popular_topics = students_topic_dist.nlargest(2).index.tolist()
    everyone_popular_topics = everyone_topic_dist.nlargest(2).index.tolist()

    # Print the results
    print("Popular topics among males:", male_popular_topics)
    print("Popular topics among females:", female_popular_topics)
    print("Popular topics among age over 20:", over_20_popular_topics)
    print("Popular topics among age under 20:", under_20_popular_topics)
    print("Popular topics among students:", students_popular_topics)
    print("Popular topics among everyone:", everyone_popular_topics)



if __name__ == '__main__':
    main()


