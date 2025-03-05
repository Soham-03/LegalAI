import time
import streamlit as st
import weaviate
import spacy
import torch
from transformers import AutoTokenizer, AutoModel
import google.generativeai as genai
import re
import os
from weaviate.auth import AuthApiKey

# Page configuration
st.set_page_config(
    page_title="Income Tax Act RAG System",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sidebar for configuration
with st.sidebar:
    st.title("Income Tax Act RAG System")
    st.write("A Retrieval-Augmented Generation system for the Income Tax Act, 1961.")
    
    # Mode tabs
    app_mode = st.radio("Select Mode", ["Process & Store", "Query & Chat"])

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# ----- THIS IS YOUR EXACT ORIGINAL CODE - NO CHANGES MADE -----

# Connect to Weaviate Cloud
auth_config = AuthApiKey(api_key="RieCYcHLyiIjEqpeXMQcEy3rcOm4HxOOLG6m")

# Initialize client with explicit authentication
client = weaviate.Client(
    url="https://rc5bwf5ru2phnsam7l5rq.c0.us-west3.gcp.weaviate.cloud", 
    auth_client_secret=auth_config
)

# Load NLP model
nlp = spacy.load("en_core_web_sm")

# Load the Legal-BERT model
tokenizer = AutoTokenizer.from_pretrained('nlpaueb/legal-bert-base-uncased')
model = AutoModel.from_pretrained('nlpaueb/legal-bert-base-uncased')

# Gemini setup
genai.configure(api_key="AIzaSyAWsKiuu8d3BtR4hpXeC_evSun00osH8Kw")
gemini_model = genai.GenerativeModel("gemini-2.0-flash")

def extract_section_references(text):
    """Extract section numbers and references from text"""
    # Pattern to find references like "Section 6", "section 9(1)", etc.
    # This improved pattern handles complex section references
    section_pattern = r'[Ss]ection\s+(\d+[A-Za-z]*)(?:\([^)]+\))?'
    matches = re.findall(section_pattern, text)
    
    # Also check for mentions of specific Income Tax Act sections
    act_section_pattern = r'(?:Income-tax\s+Act|Act)[,\s]+(\d+[A-Za-z]*)(?:\([^)]+\))?'
    act_matches = re.findall(act_section_pattern, text)
    
    # Combine and deduplicate all matches
    all_matches = matches + act_matches
    return list(set(all_matches))  # Return unique section numbers

def get_vector_embedding(text):
    """Generate embedding for text using Legal-BERT"""
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state
    attention_mask = inputs['attention_mask']
    mask = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
    sum_embeddings = torch.sum(embeddings * mask, dim=1)
    sum_mask = torch.clamp(mask.sum(dim=1), min=1e-9)
    mean_pooled = sum_embeddings / sum_mask
    return mean_pooled.squeeze(0).tolist()

def retrieve_by_section_number(section_numbers, limit=3):
    """Directly query the database for specific section numbers"""
    results = []
    for section_number in section_numbers:
        # Query using GraphQL filter to find exact section matches
        result = client.query.get(
            "IncomeTaxAct", 
            ["title", "content", "references"]
        ).with_where({
            "path": ["title"],
            "operator": "ContainsAny", 
            "valueString": [f"Section - {section_number}", f"Section - {section_number},"]
        }).do()
        
        if result["data"]["Get"]["IncomeTaxAct"]:
            results.extend(result["data"]["Get"]["IncomeTaxAct"])
            
    return results[:limit]  # Return top results up to limit

def analyze_completeness(retrieved_sections, question):
    """Analyze if retrieved sections are complete or references to other sections are needed"""
    combined_text = " ".join([section["content"] for section in retrieved_sections])
    
    # Check for references to other sections in the retrieved text
    referenced_sections = extract_section_references(combined_text)
    
    # Check if referenced sections are among the retrieved ones
    retrieved_section_ids = [section["title"].split(",")[0].strip() for section in retrieved_sections]
    missing_sections = [section for section in referenced_sections 
                        if not any(section in id for id in retrieved_section_ids)]
    
    # Check if retrieved content seems incomplete (looking for truncated content)
    is_incomplete = "..." in combined_text or combined_text.endswith(":") or "incomplete" in combined_text.lower()
    
    # Use Gemini to identify if the response is complete
    completeness_check_prompt = f"""
    Based on the question: "{question}" and the retrieved sections:
    
    {combined_text}
    
    Are there any essential sections of the Income Tax Act missing that would be needed to fully answer the question?
    Only list specific section numbers that are referenced but missing and essential (e.g., "Section 6", "Section 10(1)").
    If no essential sections are missing, reply with "COMPLETE".
    """
    
    completeness_response = gemini_model.generate_content(completeness_check_prompt)
    additional_sections = extract_section_references(completeness_response.text)
    
    if "COMPLETE" in completeness_response.text and not is_incomplete and not missing_sections:
        return True, []
    
    # Combine sections from both methods
    missing_sections = list(set(missing_sections + additional_sections))
    return False, missing_sections

def expand_topic_search(question, initial_sections):
    """Find additional relevant sections based on the question topic"""
    # Extract key legal concepts from the question
    doc = nlp(question)
    key_terms = [token.text.lower() for token in doc if token.pos_ in ["NOUN", "PROPN"] and len(token.text) > 3]
    
    # Find sections that might be related to these key terms
    additional_results = []
    if key_terms:
        # Create a search query with key terms
        search_query = " OR ".join(key_terms)
        result = client.query.get(
            "IncomeTaxAct", 
            ["title", "content", "references"]
        ).with_bm25(
            query=search_query,
            properties=["content"]
        ).with_limit(5).do()
        
        if "data" in result and "Get" in result["data"] and "IncomeTaxAct" in result["data"]["Get"]:
            additional_results = result["data"]["Get"]["IncomeTaxAct"]
    
    # Filter out sections already in initial results
    initial_titles = [section["title"] for section in initial_sections]
    filtered_results = [section for section in additional_results if section["title"] not in initial_titles]
    
    # Get the top 2 most relevant additional sections
    return filtered_results[:2]

def query_and_answer(question):
    # Step 1: Initial semantic search
    question_vector = get_vector_embedding(question)
    result = client.query.get(
        "IncomeTaxAct", 
        ["title", "content", "references"]
    ).with_near_vector({
        "vector": question_vector
    }).with_limit(3).do()
    
    retrieved = result["data"]["Get"]["IncomeTaxAct"]
    
    # Step 2: Check if the retrieved sections are complete
    is_complete, missing_sections = analyze_completeness(retrieved, question)
    
    # Step 3: If incomplete, fetch the missing explicitly referenced sections
    additional_retrieved = []
    if not is_complete and missing_sections:
        additional_sections = retrieve_by_section_number(missing_sections)
        # Add non-duplicate sections
        existing_titles = [item["title"] for item in retrieved]
        for section in additional_sections:
            if section["title"] not in existing_titles:
                additional_retrieved.append(section)
    
    # Step 4: Find more potentially relevant sections based on topic
    topic_sections = expand_topic_search(question, retrieved + additional_retrieved)
    additional_retrieved.extend(topic_sections)
    
    # Combine all retrieved sections
    all_retrieved = retrieved + additional_retrieved
    
    # Generate the answer using all retrieved sections
    prompt_template = """
You are an expert in the Income-tax Act, 1961, as amended. Your task is to provide legally precise, structured, and well-referenced answers strictly based on the provided sections of the Act. Your response must adhere to the following structure:

1. **Summary of the Answer**  
   - A concise response summarizing the key legal points.

2. **Detailed Legal Explanation**  
   - Cite all relevant sections, clauses, and sub-clauses explicitly.  
   - Clearly explain each provision with reference to its legal text.  
   - Provide step-by-step legal interpretation.

3. **Applicable Sections and References**  
   - For each referenced section, include the full section number, title, and relevant text.
   - Format each reference as follows:
     * **Section X(Y), Income-tax Act, 1961 - [Title]**
       - "[Direct quote of the relevant text]"
   - Include all subsections and clauses that you reference in your explanation.
   - Maintain proper hierarchical formatting to show the relationship between sections, subsections, and clauses.

4. **Limitations and Additional Considerations**  
   - If the provided sections do not fully address the question, explicitly state this.  
   - Suggest referring to additional provisions or legal interpretations where applicable.

---

### **Query:**  
{}

---

### **Relevant Sections from the Income-tax Act, 1961:**  
{}
{}
{}

---

**Provide your response below, ensuring adherence to the structure outlined above.**
"""

    prompt = prompt_template.format(
        question,
        "=" * 50,
        "\n\n".join([f"Section: {item['title']}\n{item['content']}" for item in all_retrieved]),
        "=" * 50
    )

    response = gemini_model.generate_content(prompt)
    return response.text, all_retrieved

# Schema setup function - for document processing
def setup_schema():
    schema = {
        "class": "IncomeTaxAct",
        "description": "Stores sections of the Income Tax Act",
        "properties": [
            {"name": "actId", "dataType": ["string"], "description": "Unique ID for the act section", "index": True},
            {"name": "title", "dataType": ["string"], "description": "Section title", "index": True},
            {"name": "content", "dataType": ["text"], "description": "Section content", "index": True},
            {"name": "references", "dataType": ["string[]"], "description": "Cross-references", "index": True},
            {"name": "publishingYear", "dataType": ["int"], "description": "Year of publication", "index": True},
        ],
        "vectorizer": "none"
    }

    if not client.schema.exists("IncomeTaxAct"):
        client.schema.create_class(schema)
        st.success("Schema created in Weaviate Cloud.")
    else:
        st.info("Schema already exists in Weaviate Cloud.")

# Text processing functions
def chunk_act_text(text):
    """Process and chunk the Income Tax Act text into sections and clauses."""
    doc = nlp(text)
    chunks = []
    current_chunk = []
    current_title = ""
    references = []
    publishing_year = 2024

    lines = text.split("\n")
    if lines:
        current_title = lines[0].strip()
        section_id_base = "6_1961_" + "_".join(current_title.split(",")[1].strip().split()[:2])

    current_clause = None
    in_subclause = False
    for line in lines[1:]:
        line = line.strip()
        if not line:
            continue

        # Detect start of a new clause or explanation
        if (line.startswith("(") and ")" in line) or line.startswith("Explanation"):
            if current_chunk and current_clause and not in_subclause:
                chunk_id = f"{section_id_base}_{len(chunks) + 1}"
                chunk_title = f"{current_title}, {current_clause}"
                chunks.append({
                    "actId": chunk_id,
                    "title": chunk_title,
                    "content": " ".join(current_chunk).strip(),
                    "references": references.copy(),
                    "publishingYear": publishing_year
                })
                current_chunk = []
                references = []
            current_clause = line.split('.')[0].strip() if '.' in line else line.strip()
            current_chunk.append(line)
            in_subclause = line.startswith("(") and ")" in line
        elif in_subclause and line.strip().startswith("("):  # Continue subclause
            current_chunk.append(line)
        else:
            current_chunk.append(line)
            in_subclause = False

        # Enhanced reference extraction across the entire content
        doc_line = nlp(line)
        for ent in doc_line.ents:
            if ent.label_ in ["LAW", "ORG"] or "section" in ent.text.lower() or "act" in ent.text.lower():
                ref = ent.text.strip()
                if ref not in references:
                    references.append(ref)

    # Save the last chunk
    if current_chunk and current_clause:
        chunk_id = f"{section_id_base}_{len(chunks) + 1}"
        chunk_title = f"{current_title}, {current_clause}"
        chunks.append({
            "actId": chunk_id,
            "title": chunk_title,
            "content": " ".join(current_chunk).strip(),
            "references": references.copy(),
            "publishingYear": publishing_year
        })

    return chunks

def generate_embeddings(chunks):
    """Generate sentence embeddings with MEAN pooling using the Legal-BERT model."""
    for chunk in chunks:
        # Tokenize the content
        inputs = tokenizer(chunk["content"], return_tensors="pt", padding=True, truncation=True, max_length=512)
        # Get model outputs
        with torch.no_grad():
            outputs = model(**inputs)
        # MEAN pooling: average the token embeddings (excluding padding tokens)
        embeddings = outputs.last_hidden_state  # Shape: (batch_size, seq_length, hidden_size)
        attention_mask = inputs['attention_mask']  # Shape: (batch_size, seq_length)
        # Expand mask to match embeddings shape
        mask = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
        # Sum embeddings along the sequence length dimension, weighted by mask
        sum_embeddings = torch.sum(embeddings * mask, dim=1)
        sum_mask = torch.clamp(mask.sum(dim=1), min=1e-9)  # Avoid division by zero
        mean_pooled = sum_embeddings / sum_mask  # Shape: (batch_size, hidden_size)
        # Convert to list for Weaviate
        chunk["vector"] = mean_pooled.squeeze(0).tolist()  # Remove batch dimension
    return chunks

def store_in_weaviate(chunks):
    """Store the chunks in Weaviate vector database."""
    with client.batch as batch:
        batch.batch_size = 100
        for chunk in chunks:
            batch.add_data_object(
                data_object={
                    "actId": chunk["actId"],
                    "title": chunk["title"],
                    "content": chunk["content"],
                    "references": chunk["references"],
                    "publishingYear": chunk["publishingYear"]
                },
                class_name="IncomeTaxAct",
                vector=chunk["vector"]
            )
    return len(chunks)

# ----- UI COMPONENTS (STREAMLIT) -----

# UI for Process & Store mode
if app_mode == "Process & Store":
    st.title("Process & Store Income Tax Act Sections")
    
    # Schema setup button
    if st.button("Setup/Check Schema"):
        setup_schema()
    
    # Text input area
    text_input = st.text_area(
        "Paste Income Tax Act Section Text",
        height=300,
        placeholder="Paste the text of an Income Tax Act section here..."
    )
    
    # Process button
    if st.button("Process & Store Text"):
        if not text_input:
            st.error("Please paste some text to process.")
        else:
            with st.spinner("Processing text..."):
                # Process text
                try:
                    chunks = chunk_act_text(text_input)
                    st.write(f"Created {len(chunks)} chunks.")
                    
                    # Display chunk preview
                    with st.expander("Preview Chunks"):
                        for i, chunk in enumerate(chunks):
                            st.markdown(f"### Chunk {i+1}: {chunk['title']}")
                            st.text(chunk['content'][:200] + "..." if len(chunk['content']) > 200 else chunk['content'])
                            st.markdown(f"References: {', '.join(chunk['references'])}")
                            st.divider()
                    
                    # Generate embeddings
                    with st.spinner("Generating embeddings..."):
                        chunks_with_embeddings = generate_embeddings(chunks)
                    
                    # Store in Weaviate
                    with st.spinner("Storing in Weaviate..."):
                        stored_count = store_in_weaviate(chunks_with_embeddings)
                    
                    st.success(f"Successfully processed and stored {stored_count} chunks in Weaviate.")
                except Exception as e:
                    st.error(f"Error during processing: {str(e)}")

# UI for Query & Chat mode
# UI for Query & Chat mode
elif app_mode == "Query & Chat":
    st.title("Income Tax Act Query System")
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask a question about the Income Tax Act..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display the user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Show assistant response with processing steps
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            
            try:
                # Show searching message
                message_placeholder.markdown("Searching for relevant sections...")
                
                # Get the answer and retrieved sections
                answer, retrieved_sections = query_and_answer(prompt)
                
                # Display retrieved sections in expandable
                sections_info = "\n\n".join([f"**{i+1}. {section['title']}**" for i, section in enumerate(retrieved_sections)])
                
                # Typing effect for the answer
                for chunk in answer.split():
                    full_response += chunk + " "
                    message_placeholder.markdown(full_response + "▌")
                    time.sleep(0.05)  # Adjust typing speed
                
                # Final display of the full answer
                message_placeholder.markdown(answer)
                
                # Expandable for retrieved sections
                st.expander("Retrieved Sections").markdown(sections_info)
                
                # Add to chat history
                st.session_state.messages.append({"role": "assistant", "content": answer})
            
            except Exception as e:
                error_message = f"Error: {str(e)}\n\nPlease try a different question or check your connection."
                message_placeholder.markdown(error_message)
                st.session_state.messages.append({"role": "assistant", "content": error_message})