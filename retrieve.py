import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download necessary NLTK data
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

def preprocess_for_retrieval(text):
    """
    Preprocess text for retrieval: remove stopwords, lemmatize, etc.
    
    Args:
        text (str): Text to preprocess
        
    Returns:
        str: Preprocessed text
    """
    # Lowercase
    text = text.lower()
    
    # Remove special characters and keep only alphanumeric and spaces
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    
    # Tokenize
    tokens = text.split()
    
    # Remove stopwords and short words
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words and len(token) > 2]
    
    # Lemmatize
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
    return ' '.join(tokens)

def extract_medical_terms(text):
    """
    Extract potential medical terms and lab test names from text
    
    Args:
        text (str): Text to extract terms from
        
    Returns:
        list: Extracted medical terms
    """
    # Common medical prefixes and suffixes
    prefixes = ['hemato', 'immuno', 'cardio', 'neuro', 'gastro', 'nephro', 'hepato', 
                'thyro', 'glyco', 'lipo', 'endo']
    suffixes = ['emia', 'itis', 'osis', 'pathy', 'gram', 'logy', 'tomy', 'ectomy']
    
    # Common lab test terms
    lab_terms = ['count', 'level', 'test', 'ratio', 'cholesterol', 'glucose', 'insulin',
                'protein', 'blood', 'urine', 'serum', 'plasma', 'rate', 'function', 'profile']
    
    # Extract terms that might be medical
    tokens = text.lower().split()
    medical_terms = []
    
    for token in tokens:
        # Check if token contains a prefix or suffix
        if any(prefix in token for prefix in prefixes) or any(token.endswith(suffix) for suffix in suffixes):
            medical_terms.append(token)
        # Check if token is a lab term
        elif token in lab_terms:
            medical_terms.append(token)
    
    return medical_terms

def create_vectorizer(processed_texts):
    """
    Creates an enhanced TF-IDF vectorizer with improved parameters and transforms the texts.

    Args:
    processed_texts (list): List of preprocessed and tokenized texts.

    Returns:
    tuple: TF-IDF vectorizer and transformed text matrix.
    """
    # Prepare texts for vectorization
    if isinstance(processed_texts[0], list):
        # If processed_texts contains lists of tokens, join them
        texts_for_vectorization = [' '.join(text) for text in processed_texts]
    else:
        # If processed_texts contains strings, use them directly
        texts_for_vectorization = processed_texts
    
    # Create enhanced TF-IDF vectorizer
    vectorizer = TfidfVectorizer(
        min_df=2,                  # Ignore terms that appear in less than 2 documents
        max_df=0.85,               # Ignore terms that appear in more than 85% of documents
        ngram_range=(1, 2),        # Include unigrams and bigrams
        sublinear_tf=True,         # Apply sublinear tf scaling (1 + log(tf))
        use_idf=True,              # Apply IDF weighting
        norm='l2',                 # Apply L2 normalization
        analyzer='word',           # Analyze by word
        tokenizer=None,            # Use scikit-learn's default tokenizer
        preprocessor=None          # Use scikit-learn's default preprocessor
    )
    
    # Transform texts to TF-IDF vectors
    X = vectorizer.fit_transform(texts_for_vectorization)
    
    return vectorizer, X

def enhance_query(query, top_docs=None):
    """
    Enhance query with medical terms and context
    
    Args:
        query (str): Original query
        top_docs (list): Optional list of top retrieved documents to extract terms from
        
    Returns:
        str: Enhanced query
    """
    enhanced_query = query.lower()
    
    # Extract medical terms from the query
    medical_terms = extract_medical_terms(query)
    
    # If we have top documents, extract medical terms from them too
    if top_docs:
        for doc in top_docs:
            doc_terms = extract_medical_terms(doc)
            medical_terms.extend(doc_terms)
    
    # Remove duplicates
    medical_terms = list(set(medical_terms))
    
    # Add weight to medical terms in the query
    if medical_terms:
        # Add medical terms to the query with repetition for emphasis
        enhanced_query += " " + " ".join(medical_terms * 2)
    
    return enhanced_query

def retrieve(query, X, vectorizer, top_k=5, use_enhanced=True, texts=None):
    """
    Retrieves the top-k most relevant texts for a given query with improved relevance.

    Args:
    query (str): Query string.
    X (matrix): TF-IDF transformed text matrix.
    vectorizer (TfidfVectorizer): TF-IDF vectorizer.
    top_k (int): Number of top results to retrieve.
    use_enhanced (bool): Whether to use query enhancement.
    texts (list): Optional list of original texts for query enhancement.

    Returns:
    list: Indices of the top-k most relevant texts.
    """
    # Preprocess the query the same way as the documents
    processed_query = preprocess_for_retrieval(query)
    
    # First pass retrieval to get initial results
    query_vec = vectorizer.transform([processed_query])
    scores = cosine_similarity(X, query_vec).flatten()
    
    # If using enhanced retrieval and we have texts
    if use_enhanced and texts:
        # Get initial top documents
        initial_top_indices = np.argsort(scores)[-min(3, len(scores)):][::-1]
        initial_top_docs = [texts[i] for i in initial_top_indices]
        
        # Enhance the query with medical terms from top docs
        enhanced_query = enhance_query(query, initial_top_docs)
        processed_enhanced_query = preprocess_for_retrieval(enhanced_query)
        
        # Second pass retrieval with enhanced query
        enhanced_query_vec = vectorizer.transform([processed_enhanced_query])
        enhanced_scores = cosine_similarity(X, enhanced_query_vec).flatten()
        
        # Combine scores (original and enhanced)
        combined_scores = scores * 0.4 + enhanced_scores * 0.6
        
        # Sort by combined scores
        top_indices = np.argsort(combined_scores)[-top_k:][::-1]
    else:
        # Use only original scores if not using enhancement
        top_indices = np.argsort(scores)[-top_k:][::-1]
    
    return top_indices.flatten()
