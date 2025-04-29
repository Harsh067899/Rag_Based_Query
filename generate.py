import nltk
from nltk.tokenize import sent_tokenize
import re
from collections import Counter
import string

# Download required NLTK data
nltk.download('punkt', quiet=True)

def extract_lab_test_info(texts):
    """
    Extract structured lab test information from retrieved texts
    
    Args:
        texts (list): List of retrieved text strings
        
    Returns:
        dict: Structured information about lab tests
    """
    # Patterns to identify lab test information
    test_pattern = r'Test:\s*([^,]+),\s*Value:\s*(\d+\.?\d*),\s*Reference Range:\s*([^,]+),\s*Out of Range:\s*(True|False)'
    
    # Results dictionary
    lab_info = {
        'tests': [],
        'abnormal_tests': [],
        'total_tests': 0,
        'abnormal_count': 0
    }
    
    # Extract lab test data from all texts
    for text in texts:
        matches = re.finditer(test_pattern, text)
        for match in matches:
            test_name = match.group(1).strip()
            test_value = float(match.group(2).strip())
            reference = match.group(3).strip()
            is_abnormal = match.group(4).strip() == 'True'
            
            test_info = {
                'name': test_name,
                'value': test_value,
                'reference': reference,
                'abnormal': is_abnormal
            }
            
            lab_info['tests'].append(test_info)
            if is_abnormal:
                lab_info['abnormal_tests'].append(test_info)
    
    # Remove duplicates by test name
    unique_tests = {}
    for test in lab_info['tests']:
        name = test['name'].lower()
        if name not in unique_tests:
            unique_tests[name] = test
    
    lab_info['tests'] = list(unique_tests.values())
    
    # Recalculate abnormal tests
    lab_info['abnormal_tests'] = [test for test in lab_info['tests'] if test['abnormal']]
    lab_info['total_tests'] = len(lab_info['tests'])
    lab_info['abnormal_count'] = len(lab_info['abnormal_tests'])
    
    return lab_info

def generate_lab_report_summary(lab_info):
    """
    Generate a summary of lab report findings
    
    Args:
        lab_info (dict): Structured lab test information
        
    Returns:
        str: Summary of lab report findings
    """
    if not lab_info['tests']:
        return "No lab test information found in the provided data."
    
    # Build summary
    summary = []
    
    # Overall stats
    summary.append(f"Found {lab_info['total_tests']} lab tests, with {lab_info['abnormal_count']} abnormal results.")
    
    # Abnormal results
    if lab_info['abnormal_tests']:
        summary.append("\nAbnormal test results:")
        for test in lab_info['abnormal_tests']:
            summary.append(f"- {test['name']}: {test['value']} (Reference range: {test['reference']})")
    else:
        summary.append("\nAll test results are within normal ranges.")
    
    return "\n".join(summary)

def extract_keywords_from_query(query):
    """
    Extract key terms from the query to guide response generation
    
    Args:
        query (str): The query text
        
    Returns:
        list: List of key terms
    """
    # Lowercase
    query = query.lower()
    
    # Specific keywords to look for
    medical_terms = [
        'normal', 'abnormal', 'range', 'test', 'high', 'low', 'elevated', 'deficient',
        'cholesterol', 'glucose', 'thyroid', 'liver', 'kidney', 'blood', 'count',
        'hemoglobin', 'platelets', 'white', 'red', 'cell', 'protein', 'lipid', 'vitamin',
        'calcium', 'potassium', 'sodium', 'iron', 'blood pressure', 'heart', 'lung'
    ]
    
    # Question types
    question_types = {
        'what': 'information',
        'which': 'selection',
        'how': 'process',
        'why': 'reason',
        'is': 'confirmation',
        'are': 'confirmation',
        'can': 'possibility',
        'should': 'recommendation',
        'would': 'hypothetical'
    }
    
    # Extract query characteristics
    keywords = []
    
    # Check for medical terms
    for term in medical_terms:
        if term in query:
            keywords.append(term)
    
    # Identify question type
    query_words = query.split()
    if query_words:
        first_word = query_words[0]
        if first_word in question_types:
            keywords.append(question_types[first_word])
    
    return keywords

def generate_response(retrieved_texts, query, max_tokens=250):
    """
    Generates a response based on the retrieved texts and query using a simple local method.

    Args:
    retrieved_texts (list): List of retrieved text strings.
    query (str): Query string.
    max_tokens (int): Maximum number of tokens for the response.

    Returns:
    str: Generated response.
    """
    # Extract lab test information
    lab_info = extract_lab_test_info(retrieved_texts)
    
    # Extract query keywords
    query_keywords = extract_keywords_from_query(query)
    
    # Check if the query is about lab results summary
    if any(term in query.lower() for term in ['summary', 'overview', 'result', 'test', 'abnormal']):
        return generate_lab_report_summary(lab_info)
    
    # Convert query to lowercase for case-insensitive matching
    query_lower = query.lower()
    
    # Check for specific query types
    if 'abnormal' in query_lower or 'out of range' in query_lower:
        # Query about abnormal results
        if lab_info['abnormal_tests']:
            abnormal_list = [f"{test['name']}: {test['value']} (Reference: {test['reference']})" 
                            for test in lab_info['abnormal_tests']]
            return f"The following tests are out of normal range:\n- " + "\n- ".join(abnormal_list)
        else:
            return "All test results are within normal ranges."
    
    elif 'normal' in query_lower:
        # Query about normal results
        normal_tests = [test for test in lab_info['tests'] if not test['abnormal']]
        if normal_tests:
            normal_list = [f"{test['name']}: {test['value']} (Reference: {test['reference']})" 
                          for test in normal_tests]
            return f"The following tests are within normal ranges:\n- " + "\n- ".join(normal_list)
        else:
            return "No tests are within normal ranges."
    
    # For general queries, use the keyword-based approach
    # Keywords to look for relevant sentences
    keywords = extract_keywords(query_lower)
    
    # Combine retrieved texts
    combined_text = "\n".join(retrieved_texts)
    
    # Split into sentences
    sentences = sent_tokenize(combined_text)
    
    # Score sentences based on keyword matches
    scored_sentences = []
    for sentence in sentences:
        score = 0
        sentence_lower = sentence.lower()
        
        for keyword in keywords:
            if keyword in sentence_lower:
                score += 1
        
        # Boost score for sentences with lab test information
        if re.search(r'test|value|range|normal|abnormal', sentence_lower):
            score += 2
            
        # Boost sentences that match query keywords
        for keyword in query_keywords:
            if keyword in sentence_lower:
                score += 3
        
        if score > 0:
            scored_sentences.append((sentence, score))
    
    # Sort sentences by score (descending)
    scored_sentences.sort(key=lambda x: x[1], reverse=True)
    
    # Check if we found any relevant sentences
    if not scored_sentences:
        # Return lab summary if no relevant sentences
        return generate_lab_report_summary(lab_info)
    
    # Take top sentences to build response (limiting by max_tokens)
    selected_sentences = []
    current_length = 0
    
    for sentence, _ in scored_sentences:
        # Approximate token count by words (not precise but good enough)
        sentence_length = len(sentence.split())
        
        if current_length + sentence_length <= max_tokens:
            selected_sentences.append(sentence)
            current_length += sentence_length
        else:
            break
    
    # Build the response
    if query_lower.startswith("what") or query_lower.startswith("which") or query_lower.startswith("how"):
        response = " ".join(selected_sentences)
    else:
        # For other queries, try to format as direct answer
        if len(selected_sentences) == 1:
            response = selected_sentences[0]
        else:
            response = "Based on the lab report:\n- " + "\n- ".join(selected_sentences)
    
    return response

def extract_keywords(query):
    """
    Extract important keywords from the query.
    
    Args:
        query (str): The query string
        
    Returns:
        list: List of keywords
    """
    # Remove common stop words
    stop_words = ["a", "an", "the", "is", "are", "in", "on", "at", "to", "for", "of", "with", 
                 "by", "about", "like", "do", "does", "what", "which", "who", "whom", "whose",
                 "when", "where", "why", "how"]
    
    # Remove punctuation
    query = query.translate(str.maketrans('', '', string.punctuation))
    
    # Tokenize the query
    words = re.findall(r'\b\w+\b', query.lower())
    
    # Filter out stop words and keep only meaningful keywords
    keywords = [word for word in words if word not in stop_words and len(word) > 2]
    
    return keywords
