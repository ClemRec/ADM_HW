import numpy as np
from math import sqrt,log
import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import words as Words
from tqdm import tqdm
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
from wordcloud import WordCloud

##------- HASH FUNCTION -------##

def hash(s, max_n_bit_hash_value):
    max_value = 1 << max_n_bit_hash_value # Define module equal to two to the thirty-second
    l = len(s) # len string
    h = 0
    for i in range(l): # For loop between 0 to len string
        h += s[i] ** (i+1) #For each character we increase "h". We compute for each character this power equation
        h %= max_value #After the power eqaution we divided the value of h by the module
    return h

class HyperLogLog:
    def __init__(self, b, max_n_bit_hash_value):
        self.max_n_bit = max_n_bit_hash_value #We set the max number of bit for the hash value obtained before
        self.b = b 
        self.m = 2**b 
        self.M = np.full(self.m, 0) #An array with size equal to m initialized with all values equal to zero.
        self.a_m = 0.7213 / (1 + 1.079/self.m) #This value was obtained from a publication reported in our references. It is a value that depends only on m
        print('b:', b, "=> Error Filter:", 1.04/sqrt(self.m), '%')
    
    def add(self, s):
        x = hash(s, max_n_bit_hash_value=self.max_n_bit) #We apply our hash functon for each string that we put in input
        j = x >> (self.max_n_bit - self.b) #We define j as the value x obtained from the hash function divided by 2 raised to the difference between the maximum number of bits set for the hash function and the variable b chosen.
        w = x & 2**(self.max_n_bit - self.b) - 1 
        # w = x & (2**(self.max_n_bit - self.b) - 1) # possibile nuova versione, da verificare le precedenze degli operatori
        self.M[j] = max(self.M[j], self.rho(w, max_length=self.max_n_bit - self.b)) #For each element in array M we upload this with the max value between M[j] and value obtain with rho function.
    
    def rho(self, n, max_length): #?
        p = len(bin(n)[2:]) # Most significant bit in n
        return max_length - p + 1
    
    def card(self):
        Z = 1/np.sum(2.**-self.M) #It is equal to the sum of 2 raised by the opposite of each element belonging to the array M
        E = self.a_m * Z * self.m ** 2 #Cardinality defined by multiplication between the calculated variable Z and the two initialized variables (a_m and m)
        if E < 5.*self.m/2.:
            V = (self.M==0).sum() #Equal to the number of element in array M equal to 0
            if V==0: #If there isn't any element in array M equal to 0
                return E 
            else:
                return self.m*log(self.m*1./V) #The new cardinality is equal to the product of m (2 ** b) and the logarithm
        return E #Return Cardinality estimate with relative error


##------- CLUSTERING -------##

def clean_text(text):
    lemmatizer = WordNetLemmatizer() #Lemmatizer 
    stop_words = set(stopwords.words("english")) #Set of all stopwords in English
    words = pos_tag(word_tokenize(text)) #Obtain for all words in text a list with tuple (the first element of tuple is the word and the second one is the tag)
    possible_tag = ["J", "R", "V", "N"] #Declare the first letter for tag. The letter N represent class for all noun. The letter J represent class for all adjective. The letter V represent class for all verbs. The letter R represent class for all adverb.  
    filtered_words = [] #New list for each rows
    for word,pos in words:
        if word.lower() not in stop_words and word.isalpha() and pos[0] in possible_tag:
            filtered_words.append(lemmatizer.lemmatize(word.lower(),"v")) #Append all word with tag in possible tag. The word is defined with lemm. The word must not be present in the stopwords
    return ' '.join(filtered_words) #Join the list to string


def save_dataset(df):
    ## Save the new dataset but before i need to remove two rows because They contain an empty text box.
    df['Text'].fillna('', inplace=True)
    for i in range(len(df)):
        if df.loc[i,"Text"] == "" :
            df.drop(i,inplace=True)
    with open("./data/CleanDatabase.csv", "w") as text_file:
        text_file.write(df.to_csv(index=False))

def tfidf_vectorizer(df):
    from sklearn.feature_extraction.text import TfidfVectorizer
    tfidf_vectorizer=TfidfVectorizer(max_features=30000, stop_words='english',use_idf=True) # Define the setting for TFIDFVectorizer
    tfidf_new=tfidf_vectorizer.fit_transform(df["Text"]) # Apply the method to a specific column in the dataset "TEXT"

    return tfidf_new

def best_compenents (tf_idf,n_components_initial,n_components_final,goal):
    #With this function i want to find the best components respect a target on the aggregate variance.
    #Since the function is very complex at the computational level, I choose ranges with respect to which to start the search for the component
    variance_init = 0 #Setting start variance
    components = n_components_initial #Setting start components equal with n_components_inital
    for k in tqdm(range(n_components_initial, n_components_final)):
        components += 1 #Increase components for each loop
        svd = TruncatedSVD(n_components=k) # Define SVD Method with components = k
        svd.fit(tf_idf) # Apply the Method to TFIDF 
        variance = float(np.cumsum(np.round(svd.explained_variance_ratio_, decimals=3)*100)[-1:]) #Search in Variance array the last element. It represents the variance for SVD Method with component = k
        if variance > variance_init: #Update variance_init
            variance_init = variance
            if variance_init >= goal: #If variable variance_init is >= respect goal I want to break the for loop and return the components respect to which the aggregate variance is greater than or equal to goal
                break
    return components

def SVDMethod(tf_idf, k):
    #SVD Method, in this case k is defined by the function best_componens
    svd = TruncatedSVD(n_components=k)
    new_matrix = svd.fit_transform(tf_idf)
    return new_matrix

class KMeans:
    def __init__(self, K, n_features, random_state=None):
        if random_state:
            np.random.seed(seed=random_state)
        self.K = K # Number of clusters
        self.f = n_features # Number of features
        self.centroids = np.random.rand(K, n_features) - 0.5 # Create the centroids using random value between -0.50 <= x < 0.50
        
    def fit(self, X, max_iter = 300, verbose=True, plot_cluster_evolution=False):
        self.centroids *= np.mean(X) # Scale the centroids to match the X
        n = X.shape[0] # n is the number of sample in X
        U = np.zeros(n) # U contains the cluster prediction
        
        inertia_values = [] # List containing the inertia values (used to plot the inertia evolution)
        changes_values = [] # List containing the number of changes occurred (used to plot the number of changes evolution)
        
        # Training loop
        for i in range(max_iter):
            U_old = np.copy(U) # U_old contains a backup of U, it's used to calculate the number of changes occurred and to check convergence
            U, inertia = self.predict(X, inertia=True)
            n_changes = np.sum(U != U_old) # calculate the number of sample that changed cluster
            
            # For each cluster we update the centroid coordinates
            for j in range(self.K): 
                filt = U == j
                if filt.any():
                    self.centroids[j] = np.mean(X[filt], axis=0) 
            
            # If verbose print evolution information
            if verbose:
                print('Iter:', i, 'Inerzia:', inertia, 'N. Changes:', n_changes)
            
            # If plot_cluster_evolution we store the inertia and the n_changes values
            if plot_cluster_evolution:
                inertia_values.append(inertia)
                changes_values.append(n_changes)
            
            
            if not n_changes: # If there are no changes it converges so break the loop
                break
            
        # If plot_cluster_evolution is true plot the inertia values and the n. changes values
        if plot_cluster_evolution:
            inertia_values = np.array(inertia_values)
            changes_values = np.array(changes_values)
            
            # Plot Inertia
            plt.figure(figsize=(16,7))
            plt.ylabel('Inertia', size=22)
            sns.lineplot(data=inertia_values)
            plt.xlabel('N. Iteration', size=20)
            plt.xticks(np.arange(inertia_values.shape[0]), np.arange(inertia_values.shape[0])+1)
            plt.show()
            
            # Plot n. changes
            plt.figure(figsize=(16,7))
            plt.ylabel('N. Changes', size=22)
            sns.lineplot(data=changes_values)
            plt.xlabel('N. Iteration', size=20)
            plt.xticks(np.arange(changes_values.shape[0]), np.arange(changes_values.shape[0])+1)
            plt.show()
        
        # then returns the inertia
        return inertia
    
    def calc_inertia(self, X):
        return self.predict(X, inertia=True)[1]
    
    def predict(self, X, inertia=False):
        n = X.shape[0] # n is the number of sample in X
        
        # Calculate the distances between each sample and each centroid
        dist = np.zeros((n, self.K)) # dist contains the distances between each sample and each centroid
        for i in range(n):
            for j in range(self.K):
                dist[i,j] = scipy.spatial.distance.euclidean(X[i], self.centroids[j]) # Calculate the euclidean distance between the sample X[i] and the centroid self.centroids[j]
        
        res = np.argmin(dist, axis=1) # Calculate, for each sample, the nearest centroid
        
        if inertia: 
            _inertia = np.sum(np.take_along_axis(dist, np.expand_dims(res, axis=-1), axis=-1)) # Calculate the inertia using the distance matrix
            return res, _inertia
        
        return res

def elbow_method(X, l, plot_result = False):
    elbow = {} #Create empty dictionary
    for k in tqdm(l): #For loop for each element in list l
        elbow_model = SK_KMeans(n_clusters=k) #KMeans algorithm from scikit-learn with number cluster = K
        elbow_model.fit_predict(X) #Fit and predict respect matrix X obtained from SVD Method
        elbow[k] = elbow_model.inertia_ #Save in dictionary key = k and value for this key equal to inertia
    #Plot the value of inertia for each k 
    if plot_result:
        plt.figure(figsize=(16,10))
        plt.plot(list(elbow.keys()), list(elbow.values()))
        plt.show()
    
    return elbow

def store_complete_dataset(df):
    #We want to save the dataset that contains for each ProductID the cluster associated through the join between the initial dataset
    #and the one we worked with in Clustering (on which there is Groupby with respect to ProductID) containing
    #ProductID column, Clean Text and number of Cluster.
    product_cluster = df.set_index('ProductId')['Cluster']
    reviews = pd.read_csv('./data/Reviews.csv')
    rev_complete = reviews.join(product_cluster, on='ProductId')
    with open("./data/Complete_dataset.csv", "w", encoding="utf-8") as text_file:
        text_file.write(rev_complete.to_csv(index=False))

def show_word_cloud(df):
    wordlist = Words.words() #List of all word from nltk library
    cluster_text = df.groupby('Cluster').agg({'Text': ' '.join}) #We take a groupby respect Cluster and unite the text for each cluster
    for i in cluster_text.index:
        txt = cluster_text.loc[i].Text
        
        # Count the words in the text for each cluster
        cv = CountVectorizer(min_df=0, stop_words="english", max_features=200)
        counts = cv.fit_transform([txt]).toarray().ravel()
        words = np.array(cv.get_feature_names())
        
        # Filter the words: We take only words in wordlist and with tag equal to NN (Noun, common, singular) and NNS (Noun, common, plural)
        tag = np.array(pos_tag(words))[:,1]
        acceptable_word = [True if word in wordlist else False for word in words]
        word_filter = np.logical_and(np.logical_or(tag == 'NN', tag == 'NNS'), acceptable_word)
        
        # Find the top n_top most frequent words in the text
        n_top = 50 #Set parameter)
        top_word = np.array(words)[np.argsort(counts*word_filter)[-n_top:]].tolist() #The array contains the most famous words sorted by the value obtained with CountVectorizer for each word
        top_count = counts[np.argsort(counts*word_filter)[-n_top:]].tolist() #For each top_word i put in this array the value from CountVectorizer
        word_freq= dict(zip(top_word, top_count)) #Create a dictionary with key the top_word and value for each keys the top_count
        
        # Plot the wordcloud
        plt.figure(figsize=(10,10))
        wc = WordCloud(width=1000,height=1000).generate_from_frequencies(word_freq)
        plt.imshow(wc)
