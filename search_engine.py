# Import libraries
from bs4 import BeautifulSoup 
import requests                
import json                 
import nltk                    
import string
import sys                 
from nltk.corpus import stopwords     
from nltk.stem import WordNetLemmatizer 
from collections import defaultdict   
import ipywidgets as widgets 
from IPython.display import display
import numpy as np            
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.metrics.pairwise import cosine_similarity      
from rank_bm25 import BM25Okapi
from sklearn.metrics import precision_score, recall_score, f1_score  
from IPython.display import display, Markdown               
import pandas as pd            

# Download NLTK datasets
nltk.download('punkt')      # Tokenizer models
nltk.download('stopwords')  # List of stopwords
nltk.download('wordnet')    # For lemmatization

# Initialize stopwords and lemmatizer
stop_words = set(stopwords.words('english'))  # Set of stopwords
lemmatizer = WordNetLemmatizer()              # Lemmatizer for reducing words to base forms


# Function to extract paragraphs from each page
def get_paragraphs(soup, visited_paragraphs, link):
    # Initialize list to store paragraphs
    paragraphs = []  

    # Remove tags 'sup' for superscripts and 'reflist' for references and their content
    for sup in soup.find_all(['sup', 'reflist']):
        sup.decompose()  

    # Find all <p> tags to extract text content
    for p in soup.find_all('p'):
        # Extract text from each paragraph
        text = p.get_text()  
        
        # Filter out empty paragraphs and those containing the term 'displaystyle' to avoid mathematical functions
        if text and 'displaystyle' not in text.lower():
            # Calculate the number of words in the paragraph
            word_count = len(text.split())  
            
            # Only include paragraphs with word count between 50 and 100 and avoid duplicates
            if 50 <= word_count <= 100 and text not in visited_paragraphs:
                # Store paragraph with source link
                paragraphs.append({"text": text, "link": link}) 
                # Mark the paragraph as visited to avoid repetition
                visited_paragraphs.add(text)  
  
    # Return the list of filtered paragraphs         
    return paragraphs  


# Function to extract links from each page
def get_links(soup):
    # Base wikipedia URL
    https = 'https://en.wikipedia.org'  
    # List to store valid links
    links = []  

    # Find all anchor tags with 'href' attribute
    for link in soup.find_all('a', href = True):
        # Extract link reference
        url = link.get('href') 

        # Check if the link is a wikipedia aticle and filter out irrelevant links
        if url.startswith('/wiki/') and not any(
            url.startswith(f'/wiki/{keyword}')
            for keyword in ['Wikipedia', 'Help', 'Special', 'Portal', 'Talk', 'Category', 'File', 'Main_Page']):
            # Construct full wikipedia URL
            full_url = f"{https}{url}"
            
            # Avoid adding duplicate links
            if full_url not in links:
                links.append(full_url)
        
        # Limit the number of links collected
        if len(links) >= 70:
            break 

    # Return the list of valid links
    return links  


# Function to get and extract paragraphs from each link
def links_within_paragraphs(link, visited_links, visited_paragraphs):
    # Skip the link and return an empty list if it has already been processed
    if link in visited_links:
        return []

    try:
        # Send get request to link and raise error if the request was unsuccessful
        response = requests.get(link)
        response.raise_for_status()
        
        # Parse the page content
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Mark this link as visited
        visited_links.add(link)
        
        # Extract and return paragraphs from the page
        return get_paragraphs(soup, visited_paragraphs, link)
        
    # Exception for request errors
    except requests.RequestException as e:
        print(f"Error retrieving links: {e}")
        # Return an empty list if an error occurs
        return []  


# Function for text cleaning and tokenization
def clean_text(paragraph):
    # Extract text from the paragraph dictionary since it is saved as text:... link:...
    text = paragraph['text']

    # Tokenize text into individual words and convert to lowercase for better search results
    tokens = nltk.word_tokenize(text.lower())

    # List to store cleaned tokens
    cleaned_tokens = []

    # Remove punctuation and non alphanumeric tokens
    for token in tokens:
        if token not in string.punctuation and token.isalnum():
            cleaned_tokens.append(token)

    # List to store tokens after stopword removal
    filtered_tokens = []

    # Remove stopwords from the tokenized text
    for token in cleaned_tokens:
        if token not in stop_words:
            filtered_tokens.append(token)

    # Initialize a list to store lemmatized tokens
    lemmatized_tokens = []

    # Lemmatize the tokens
    for token in filtered_tokens:
        lemmatized_tokens.append(lemmatizer.lemmatize(token))

    # Return lemmatized tokens
    return lemmatized_tokens


# Function to build an inverted index from the collected paragraphs
def build_inverted_index(paragraphs):
    # Defaultdict where keys are tokens and values are sets of paragraph IDs
    inverted_index = defaultdict(set)
    
    # Look through each paragraph and assign a unique ID / index
    for paragraph_id, paragraph in enumerate(paragraphs):
        # Clean and tokenize the paragraph text
        tokens = clean_text(paragraph)
        
        # Add each token to the inverted index with its associated paragraph ID
        for token in tokens:
            inverted_index[token].add(paragraph_id)

    sample_index = list(inverted_index.items())[:6]
    inverted_df = pd.DataFrame(sample_index, columns = ['Token', 'Paragraph IDs'])
    display(Markdown("Example to show inverted index."))
    pd.set_option('display.max_colwidth', None)
    display(inverted_df)
    display(Markdown("<br><br>"))
             
    # Return the inverted index that was created
    return inverted_index


# Function to convert the boolean query from infix notation to postfix notation (Reverse Polish Notation)
def infix_to_postfix(query):
    # Define operator precedence (higher value means higher precedence)
    precedence = {'NOT': 3, 'AND': 2, 'OR': 1}
    
    # Lists for output and operator stack
    output = []  
    operators = []

    # Split the query into individual tokens
    tokens = query.split()
    
    # Process each token in the query
    for token in tokens:
        # If the token is an operator handle based on precedence
        if token in precedence:
            # Pop operators with higher or equal precedence from the stack
            while operators and precedence.get(operators[-1], 0) >= precedence[token]:
                output.append(operators.pop())
            operators.append(token)  # Push the current operator to the stack

        # If the token is left parenthesis push it onto the stack
        elif token == '(':
            operators.append(token)

        # If the token is right parenthesis pop until the matching left parenthesis
        elif token == ')':
            while operators and operators[-1] != '(':
                output.append(operators.pop())
            # Remove left parenthesis from stack
            operators.pop()  

        # If the token is a word lemmatize and add to output
        else:
            token = lemmatizer.lemmatize(token.lower())
            output.append(token)

    # Pop any remaining operators from the stack and append them to output
    while operators:
        output.append(operators.pop())

    # Return the query in postfix notation (RPN)
    return output


# Function to evaluate the boolean query in postfix notation using the inverted index
def evaluate_postfix(postfix_tokens, inverted_index, num_paragraphs):
    # Stack for evaluating the postfix expression
    stack = []  
    # For handling NOT operations
    all_paragraphs = set(range(num_paragraphs))  

    # Look through each token in the expression and preform nessesary operations
    for token in postfix_tokens:
        if token == 'AND':
            # Pop the top two sets
            right = stack.pop()  
            left = stack.pop()
            # Push the result of addition to stack
            stack.append(left & right)  

        elif token == 'OR':
            right = stack.pop()
            left = stack.pop()
            # Push the result of union to stack
            stack.append(left | right)  

        elif token == 'NOT':
            operand = stack.pop()
            # Push ducuments that are not in list to stack
            stack.append(all_paragraphs - operand)  

        # If token is a search term retrieve the matching paragraph IDs from the inverted index
        else:
            # Push matching paragraph IDs to stack
            stack.append(inverted_index.get(token, set())) 

    # Return the final result if it exists or empty set
    if stack:
        return stack.pop()
    else:
        return set()


# Function to calculate the TF-IDF matrix for the resulting paragraphs of the query
def tf_idf(results, cleaned_paragraphs):
    # Return nothing if there are no results
    if not results:
        return None, [], []

    # Lists to store the filtered paragraphs and their IDs
    filtered_paragraphs = []
    filtered_ids = []

    # Extract the text and IDs of paragraphs that match the query results
    for paragraph_id in results:
        filtered_paragraphs.append(cleaned_paragraphs[paragraph_id]['text'])
        filtered_ids.append(paragraph_id)

    # Initialize and compute the TF-IDF matrix of the paragraphs
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(filtered_paragraphs)

    # Return the TF-IDF matrix, paragraphs, their IDs and TF-IDF vectorizer
    return tfidf_matrix, filtered_paragraphs, filtered_ids, vectorizer


# Function to calculate the TF-IDF matrix for resulting paragraphs of the query
def tf_idf(results, cleaned_paragraphs):
    # Return nothing if there are no results
    if not results:
        return None, [], []

    # Lists to store the filtered paragraphs and their IDs
    filtered_paragraphs = []
    filtered_ids = []

    # Extract the text and IDs of paragraphs that match the query results
    for paragraph_id in results:
        filtered_paragraphs.append(cleaned_paragraphs[paragraph_id]['text'])
        filtered_ids.append(paragraph_id)

    # Initialize and compute the TF-IDF matrix of the paragraphs
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(filtered_paragraphs)

    # Return the TF-IDF matrix, paragraphs, their IDs and TF-IDF vectorizer
    return tfidf_matrix, filtered_paragraphs, filtered_ids, vectorizer


# Function to rank paragraphs using the Vector Space Model and cosine similarity
def vector_space_model(cleaned_query, tfidf_matrix, original_paragraphs, filtered_ids, vectorizer):
    # Convert the query list into a string
    cleaned_query = ' '.join(cleaned_query)

    # Transform the query into a TF-IDF vector using the TF-IDF vectorizer
    query_vector = vectorizer.transform([cleaned_query])

    # Compute the cosine similarity between the query and the TF-IDF matrix
    cosine_similarities = cosine_similarity(query_vector, tfidf_matrix)[0]

    # To store the paragraph with biggest score for each link
    link_top_scores = {}

    # Look through each paragraph and its similarity score
    for paragraph_id, score in zip(filtered_ids, cosine_similarities):
        # Retrieve the original paragraph data to print
        original = original_paragraphs[paragraph_id]
        # Extract the link of current paragraph
        link = original['link']  

        # If the link is new or the score is higher than the existing one update the record
        if link not in link_top_scores or score > link_top_scores[link]['score']:
            link_top_scores[link] = {
                'paragraph_id': paragraph_id,
                'link': link,
                'text': original['text'],
                'score': score
            }

    # Sort the results in descending order
    sorted_scores = sorted(
        link_top_scores.values(),
        key = lambda x: x['score'],
        reverse = True
    )

    # Return the list of highest ranked paragraphs 
    return sorted_scores


# Function to rank paragraphs using the Okapi BM25 algorithm
def okapi_bm25(cleaned_query, filtered_paragraphs, filtered_ids, original_paragraphs):
    # Tokenize the filtered paragraphs
    tokenized_paragraphs = [paragraph.split(" ") for paragraph in filtered_paragraphs]

    # Initialize BM25 Okapi and fit it on the tokenized paragraphs
    bm25 = BM25Okapi(tokenized_paragraphs)

    # Compute the BM25 scores for the query
    bm25_scores = bm25.get_scores(cleaned_query)

    # To keep track of the highest ranked paragraph for each link
    link_top_scores = {}

    # Look through the filtered paragraph IDs and their BM25 scores
    for paragraph_id, score in zip(filtered_ids, bm25_scores):
        # Retrieve original paragraph data for printing
        original = original_paragraphs[paragraph_id]
        link = original['link']

        # Store the paragraph only if it has the highest score for this link
        if link not in link_top_scores or score > link_top_scores[link]['score']:
            link_top_scores[link] = {
                'paragraph_id': paragraph_id,
                'link': link,
                'text': original['text'],
                'score': score
            }

    # Sort the results descending order
    sorted_scores = sorted(
        link_top_scores.values(),
        key = lambda x: x['score'],
        reverse = True
    )

    # Return the highest ranked paragraphs
    return sorted_scores


# Interactive search engine interface
def search_engine(inverted_index, original_paragraphs, cleaned_paragraphs):
    # Widget for selecting the ranking algorithm
    toggle_buttons = widgets.ToggleButtons(
        options=['Boolean retrieval', 'Vector Space Model', 'Okapi BM25'],
        description='Select ranking algorithm'
    )
    
    # Space widget for easier viewing
    space = widgets.HTML(value = '<br>')

    # Text input widget for entering search queries
    input_text = widgets.Text(
        placeholder = 'Input search query here',
        layout = widgets.Layout(width = '60%')
    )
    
    # Button widget for triggering the search
    search_button = widgets.Button(
        description = 'Search',
        button_style = 'primary'
    )
    
    # Output widget for displaying search results
    output = widgets.Output()

    # Internal function to handle the search when the button is clicked
    def search(b):
        # Clear previous results
        output.clear_output()  
        # Get the selected ranking algorithm
        algorithm = toggle_buttons.value  
        # Get query that user entered
        query = input_text.value  

        # Convert the query to postfix and evaluate using the inverted index
        postfix_query = infix_to_postfix(query)
        results = evaluate_postfix(postfix_query, inverted_index, len(cleaned_paragraphs))

        # If no results match the query
        if not results:
            with output:    
                print(f"No results found for '{query}' using {algorithm}.")
            return
        
        # Apply TF-IDF transformation on the resulting paragraphs
        tfidf_matrix, filtered_paragraphs, filtered_ids, vectorizer = tf_idf(results, cleaned_paragraphs)
        # Clean the query for ranking
        cleaned_query = clean_text({"text": query})  

        # Display search results based on the selected ranking algorithm
        with output:
            if filtered_paragraphs:
                print(f"Total matching paragraphs: {len(filtered_ids)}\n")

                # Display the first paragraph per link (Boolean retrieval)
                if algorithm == 'Boolean retrieval':
                    displayed_links = set()
                    for i, paragraph_id in enumerate(filtered_ids):
                        original = original_paragraphs[paragraph_id]
                        if original['link'] not in displayed_links:
                            displayed_links.add(original['link'])
                            print(f"Link: {original['link']}\n{original['text']}\n")
                    print(f"Total links shown: {len(displayed_links)}\n")
                
                # Rank paragraphs using cosine similarity (VSM)
                elif algorithm == 'Vector Space Model':
                    ranked_results = vector_space_model(cleaned_query, tfidf_matrix, original_paragraphs, filtered_ids, vectorizer)
                    displayed_links = set()
                    for result in ranked_results:
                        displayed_links.add(result['link'])
                        print(f"Link: {result['link']}\n(Score: {result['score']:.3f})\n{result['text']}\n")
                    print(f"Total links shown: {len(displayed_links)}\n")
                
                # Rank paragraphs using BM25 scoring (Okapi BM25)
                elif algorithm == 'Okapi BM25':
                    ranked_results = okapi_bm25(cleaned_query, filtered_paragraphs, filtered_ids, original_paragraphs)
                    displayed_links = set()
                    for result in ranked_results:
                        displayed_links.add(result['link'])
                        print(f"Link: {result['link']}\n(Score: {result['score']:.3f})\n{result['text']}\n")
                    print(f"Total links shown: {len(displayed_links)}\n")
               
    # Connect the search button to the search function
    search_button.on_click(search)

    # Display widgets
    display(widgets.VBox([toggle_buttons, space, input_text, search_button, output]))


# Function to store collected links in a JSON file
def store_links(links):
    try:
        # Open the file in write mode and insert the list of links
        with open('wikipedia_collected_urls.json', 'w') as file:
            json.dump(links, file)

    except Exception as e:
        # Error handling
        print(f"____Error saving links____\n{e}")


# Function to store paragraphs in a JSON file
def store_paragraphs(paragraphs, filename):
    try:
        # Open the file in write mode and insert the paragraphs list
        with open(filename, 'w') as file:
            json.dump(paragraphs, file)

    except Exception as e:
        print(f"____Error saving paragraphs____\n{e}")


# Function to store an inverted index in a JSON file
def store_inverted_index(inverted_index):
    try:
        # Convert sets to lists to save in JSON file
        serializable_index = {}
        for term, paragraph_ids in inverted_index.items():
            serializable_index[term] = list(paragraph_ids)

        # Save the converted index to the file
        with open('inverted_index.json', 'w') as file:
            json.dump(serializable_index, file)

    except Exception as e:
        print(f"____Error saving inverted index____\n{e}")


# Function to scrape wikipedia data, extract links, paragraphs and store results
def get_wiki(wiki_url):
    try:
        # Send request to the specified wikipedia URL
        response = requests.get(wiki_url)
        # Raise an exception for HTTP errors
        response.raise_for_status()
        
        # Parse the HTML content using
        soup = BeautifulSoup(response.text, 'html.parser')
        
    # Handle any exceptions during the HTTP request and exit function
    except requests.RequestException as e:
        print(f"____Request failed____\n{e}\n")
        return  

    # Collect and store links from the main page
    links = get_links(soup)
    store_links(links)

    # Sets for tracking visited pages and paragraphs
    visited_links = set()
    visited_paragraphs = set()

    # Collect and clean paragraphs from the main page
    original_paragraphs = get_paragraphs(soup, visited_paragraphs, wiki_url)
    cleaned_paragraphs = []

    # Clean the collected paragraphs using text preprocessing
    for paragraph in original_paragraphs:
        clean_paragraph = ' '.join(clean_text(paragraph))
        cleaned_paragraphs.append({"text": clean_paragraph, "link": paragraph["link"]})

    # Mark the main page as visited
    visited_links.add(wiki_url)

    # Progress bar to show data being scraped
    progress = widgets.IntProgress(
        value = 0,
        min = 0,
        max = len(links),
        description = 'Scraping Data'
    )
    
    # Add space for easier reading and display progress bar
    space = widgets.HTML(value = '<br>')
    display(widgets.VBox([progress, space]))

    # Get paragraphs from each link and avoid re visiting links
    for i, link in enumerate(links):
        # Get paragraphs from the current link
        link_paragraphs = links_within_paragraphs(link, visited_links, visited_paragraphs)

        # Extend list of original paragraphs with the new data
        original_paragraphs.extend(link_paragraphs)

        # Clean and store paragraphs from current link
        for paragraph in link_paragraphs:
            clean_paragraph = ' '.join(clean_text(paragraph))
            cleaned_paragraphs.append({"text": clean_paragraph, "link": paragraph["link"]})

        # Update progress bar
        progress.value = i + 1

    # Store collected paragraphs in JSON files
    store_paragraphs(original_paragraphs, 'wikipedia_paragraphs.json')
    store_paragraphs(cleaned_paragraphs, 'wikipedia_paragraphs_cleaned.json')

    # Build inverted index from cleaned paragraphs
    inverted_index = build_inverted_index(cleaned_paragraphs)

    # Store generated inverted index
    store_inverted_index(inverted_index)


if __name__ == "__main__":
    try:
        # Define the starding wikipedia URL to scrape
        wiki_url = 'https://en.wikipedia.org/wiki/Data_analysis'
        
        # Get and process data from page
        get_wiki(wiki_url)
        
        # Load the collected links from the JSON file
        with open('wikipedia_collected_urls.json', 'r') as file:
            links = json.load(file)
            
        # Load the original paragraphs
        with open('wikipedia_paragraphs.json', 'r') as file:
            original_paragraphs = json.load(file)
    
        # Load the cleaned paragraphs 
        with open('wikipedia_paragraphs_cleaned.json', 'r') as file:
            cleaned_paragraphs = json.load(file)
    
        # Load the inverted index 
        with open('inverted_index.json', 'r') as file:
            data = json.load(file)
            inverted_index = {}
            
            # Convert the inverted index data from lists to sets
            for term, paragraph_ids in data.items():
                inverted_index[term] = set(paragraph_ids)
    
        # Launch the search engine interface using the collected data
        search_engine(inverted_index, original_paragraphs, cleaned_paragraphs)

    # Handle missing files error
    except FileNotFoundError as e:
        print(f"Error: {e}")
