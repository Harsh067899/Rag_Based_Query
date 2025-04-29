import os
import sys
import json
import argparse
from datetime import datetime
import pandas as pd
from tabulate import tabulate

# Import RAG components
from retrieve import create_vectorizer, retrieve
from generate import generate_response
from preprocess import preprocess_text

def format_lab_report(extracted_text_file, output_format="markdown"):
    """
    Format extracted OCR text into a more readable structure
    
    Args:
        extracted_text_file: Path to extracted text file
        output_format: Format to output (text, markdown, html)
    
    Returns:
        Formatted text
    """
    # Read the extracted text file
    with open(extracted_text_file, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # Preprocess to fix common OCR errors
    text = text.replace('€', 'c')
    text = text.replace('_', '-')
    text = text.replace('@', 'a')
    
    # Split the text into lines for processing
    lines = text.strip().split('\n')
    
    # Extract the header, lab info, and test results
    header_lines = []
    lab_info_lines = []
    tests = []
    current_section = "header"
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Simple heuristic to detect sections
        if "BIOCHEMISTRY" in line or "TEST" in line:
            current_section = "tests"
            continue
            
        if current_section == "header" and any(keyword in line.upper() for keyword in ["HOSPITAL", "LAB", "DIAGNOSTIC", "PATHOLOGY"]):
            header_lines.append(line)
        elif current_section == "header":
            lab_info_lines.append(line)
        elif current_section == "tests" and any(c.isdigit() for c in line):
            tests.append(line)
    
    # Format header
    formatted_header = "\n".join(header_lines)
    
    # Format lab info
    formatted_lab_info = "\n".join(lab_info_lines)
    
    # Process test results to extract structured data
    structured_tests = []
    
    for test_line in tests:
        parts = test_line.split()
        if len(parts) < 2:
            continue
            
        # Try to find test name and value
        test_name = ""
        test_value = ""
        reference_range = ""
        
        # Simple heuristic: first parts are name, first number encountered is value,
        # and remaining parts after value might be reference range
        name_parts = []
        found_value = False
        
        for part in parts:
            if not found_value and part.replace('.', '').isdigit():
                found_value = True
                test_value = part
            elif found_value:
                reference_range += part + " "
            else:
                name_parts.append(part)
                
        test_name = " ".join(name_parts)
        
        if test_name and test_value:
            structured_tests.append({
                "test_name": test_name,
                "test_value": test_value,
                "reference_range": reference_range.strip()
            })
    
    # Create formatted output based on requested format
    if output_format == "markdown":
        output = f"# {formatted_header}\n\n{formatted_lab_info}\n\n## Test Results\n\n"
        
        if structured_tests:
            # Create markdown table
            output += "| Test | Value | Reference Range |\n"
            output += "|------|-------|----------------|\n"
            
            for test in structured_tests:
                output += f"| {test['test_name']} | {test['test_value']} | {test['reference_range']} |\n"
        else:
            output += "No structured test results could be extracted."
            
    elif output_format == "html":
        output = f"<h1>{formatted_header}</h1>\n<p>{formatted_lab_info}</p>\n<h2>Test Results</h2>\n"
        
        if structured_tests:
            # Create HTML table
            output += "<table border='1'>\n<tr><th>Test</th><th>Value</th><th>Reference Range</th></tr>\n"
            
            for test in structured_tests:
                output += f"<tr><td>{test['test_name']}</td><td>{test['test_value']}</td>"
                output += f"<td>{test['reference_range']}</td></tr>\n"
                
            output += "</table>"
        else:
            output += "<p>No structured test results could be extracted.</p>"
    else:  # Plain text format
        output = f"{formatted_header}\n\n{formatted_lab_info}\n\nTest Results:\n\n"
        
        if structured_tests:
            # Use tabulate for nice text tables
            table_data = [[test['test_name'], test['test_value'], test['reference_range']] 
                         for test in structured_tests]
            output += tabulate(table_data, headers=["Test", "Value", "Reference Range"])
        else:
            output += "No structured test results could be extracted."
    
    return output, structured_tests

def query_rag_with_lab_data(structured_tests, query=None):
    """
    Query the RAG system with lab test data
    
    Args:
        structured_tests: List of structured test dictionaries
        query: Optional user query about the lab results
    
    Returns:
        RAG generated response
    """
    # Convert structured tests to document format for RAG
    lab_document = ""
    for test in structured_tests:
        lab_document += f"Test: {test['test_name']}, Value: {test['test_value']}, "
        lab_document += f"Reference Range: {test['reference_range']}\n"
    
    # Preprocess the document
    processed_text = preprocess_text([lab_document])[0]
    
    # Create vectorizer with just this document
    vectorizer, X = create_vectorizer([processed_text])
    
    # If no query provided, use a default query
    if not query:
        query = "Summarize the lab results and identify any abnormal values."
    
    # Retrieve context
    top_indices = retrieve(query, X, vectorizer, top_k=1)
    context = [lab_document]  # We only have one document
    
    # Generate response
    response = generate_response(context, query)
    
    return response

def enhance_ocr_output(results_dir, query=None, output_format="markdown"):
    """
    Enhance OCR output by formatting and analyzing with RAG
    
    Args:
        results_dir: Directory containing OCR results
        query: Optional query for RAG analysis
        output_format: Format for output (text, markdown, html)
    
    Returns:
        Tuple of (formatted_output, rag_response)
    """
    # Find extracted text file
    extracted_text_file = os.path.join(results_dir, "extracted_text.txt")
    if not os.path.exists(extracted_text_file):
        return "Error: Extracted text file not found", None
    
    # Format the lab report
    formatted_output, structured_tests = format_lab_report(extracted_text_file, output_format)
    
    # Query RAG system if tests were extracted
    rag_response = None
    if structured_tests:
        rag_response = query_rag_with_lab_data(structured_tests, query)
    
    # Create enhanced output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"enhanced_results_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save formatted output
    extension = "md" if output_format == "markdown" else "html" if output_format == "html" else "txt"
    with open(os.path.join(output_dir, f"formatted_report.{extension}"), 'w', encoding='utf-8') as f:
        f.write(formatted_output)
    
    # Save RAG response if available
    if rag_response:
        with open(os.path.join(output_dir, "rag_analysis.txt"), 'w', encoding='utf-8') as f:
            f.write(rag_response)
    
    return formatted_output, rag_response

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Enhance OCR output with formatting and RAG analysis")
    parser.add_argument("--results-dir", required=True, help="Directory containing OCR results")
    parser.add_argument("--query", help="Query for RAG analysis (optional)")
    parser.add_argument("--format", choices=["text", "markdown", "html"], default="text", 
                        help="Output format (default: text)")
    args = parser.parse_args()
    
    formatted_output, rag_response = enhance_ocr_output(args.results_dir, args.query, args.format)
    
    print("\n" + "="*50)
    print("FORMATTED LAB REPORT:")
    print("="*50)
    print(formatted_output)
    
    if rag_response:
        print("\n" + "="*50)
        print("RAG ANALYSIS:")
        print("="*50)
        print(rag_response)
        
    print("\nEnhanced results saved to directory with timestamp.") 