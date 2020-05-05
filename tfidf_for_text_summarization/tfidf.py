# Implementing Term Frequency (TF) - Inverse Document Frequency (IDF).
# - TF(term) = num appearances in document / total num terms in document. 
# - A measure of how common a word is.
#
# - IDF(term) = log_e(num documents / number of documents with term in it)
# - How rare a term is.  
#
# - TFIDF(term) = TF(term) * IDF(term)            
# - A measure of how important a word is to a document in a collection.

import re
import numpy as np

from nltk import sent_tokenize
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Prepares data for analysis. Removes punctuations & stopwords, & lemmatizes.
def prep_documents(documents):

    # Getting list of stopwords
    stop_words = list(set(stopwords.words('english')))

    # Lemmatizer.
    lem = WordNetLemmatizer()

    prepped_documents = []
    
    for doc in documents:  

        # Normalizing: convert to lowercase and remove punctuation, newlines, digits . 
        doc = re.sub(r'[^\w\s]', '', doc).lower()
        doc = re.sub(r'[\n\d]', '', doc)

        # Tokenizing into words. 
        words = word_tokenize(doc)

        # lemmatizing, removing stopwords.
        updated_doc = []
        for word in words:
        	if word not in stop_words:
        		updated_doc.append(lem.lemmatize(word))

        # Adding updated document to list.
        prepped_documents.append(updated_doc)

    return prepped_documents


# Returns the number of times each word appears in each document.
# Dictionary[word] = [frequency in doc1, frequency in doc2, frequency in doc3...]
def get_word_frequency(documents):

    frequency = {}
    
    # Go through each document.
    for doc_num in range(len(documents)):

        # Go through each word in document.
    	for word in documents[doc_num]:

    		# Word has appeared but its the first time in this document.
    		if word in frequency and len(frequency[word]) == doc_num:
    			frequency[word].append(documents[doc_num].count(word))

    		# Word hasn't appeared yet. Create array pad previous doc values with zero. 
    		elif word not in frequency:
    			frequency[word] = []
    			for i in range(doc_num):
    				frequency[word].append(0)

    			frequency[word].append(documents[doc_num].count(word))

    	# Add a zero to all the word counts that didn't appear in the current document. 
    	for word in frequency:
    		if len(frequency[word]) < doc_num + 1:
    			frequency[word].append(0)

    return frequency


# Returns the number of words in each document, in array format where index zero is represents first document. 
def get_document_sizes(documents):

	sizes = []

	for doc in documents:
		sizes.append(len(doc))

	return sizes


# Returns the number of documents each word appears in.
# num_docs_per_word[word] = 1 or 2...
def get_num_documents_per_word(word_frequency): 

    num_docs_per_word = {}

    # Iterate through each word.
    for word in word_frequency:

        # Iterate through each document and tally the results.
    	num_docs_per_word[word] = 0
    	for doc in word_frequency[word]:

    		# If the word appears in the document, increment count.
    		if doc > 0:
    			num_docs_per_word[word] += 1

    return num_docs_per_word


# Calculates tfidf for each word-document pair. 
def calculate_tfidf(word_frequency, documents):

    # How many documents each word appears in and how many words in each document. 
	num_docs_per_word = get_num_documents_per_word(word_frequency)
	document_sizes = get_document_sizes(documents)

	# Calculating TF.
	tf = {}
	for word in word_frequency:
		tf[word] = np.array(word_frequency[word]) / np.array(document_sizes)

	# Calculating IDF.
	idf = {}
	for word in word_frequency:
		idf[word] = np.log( len(document_sizes) / num_docs_per_word[word])

	# Calculating TD-IDF.
	tfidf = {}
	for word in word_frequency:
		tfidf[word] = tf[word] * idf[word]

	return tfidf


# Calculates Document Rank ie finding the most important documents by summing TFIDF scores of their words. 
def calculate_document_rank(tfidf, documents):

	scores = [0] * len(documents)

    # Iterating through each document. 
	for i in range(len(documents)):

		# Summing tfidf scores for individual words therefore representing the document. 
		for word in documents[i]:
			scores[i] += tfidf[word][i]

	# Sorting documents based off score.
	ranked = [x for _, x in sorted(zip(scores, documents), reverse=True)]
	scores.sort(reverse=True)

	for i in range(len(scores)):
		print(f'Document Score: {scores[i]}. Document: {ranked[i]}')


def main():

	# Six documents to analyze.
	d1 = "born December 30, 1984 is an American professional basketball player for the Los Angeles Lakers of the National Basketball Association (NBA)."
	d2 = "He is widely considered to be one of the greatest basketball players in NBA history, and discussions ranking him as the greatest basketball player of all time have often been subject to significant debate, with frequent comparisons to Michael Jordan.[1]"
	d3 = "James's teams have played in eight consecutive NBA Finals (2011–2018 seasons) between the Miami Heat and Cleveland Cavaliers."
	d4 = "His accomplishments include three NBA championships, four NBA Most Valuable Player (MVP) Awards, three Finals MVP Awards, and two Olympic gold medals." 
	d5 = "James holds the all-time record for playoffs points, is third in all-time points, and eighth in all-time assists." 
	d6 = "James was selected to the All-NBA First Team twelve times (all-time record), made he All-Defensive First Team five times, and has played in sixteen All-Star Games, in which he was selected All-Star MVP three times."
	
	# Cleans documents. 
	documents = prep_documents([d1, d2, d3, d4, d5, d6])

	# Calculating TFIDF.	
	word_frequency = get_word_frequency(documents)
	tfidf = calculate_tfidf(word_frequency, documents)

	# Ranking documents in importance.
	calculate_document_rank(tfidf, documents)


if __name__ == "__main__":
	main()






