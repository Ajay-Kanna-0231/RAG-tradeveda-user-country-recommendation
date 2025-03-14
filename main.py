from flask import Flask, render_template, request, jsonify, session
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from serpapi import GoogleSearch
from datetime import datetime
import time
import re
import threading
from langchain_core.messages import HumanMessage, AIMessage
from werkzeug.utils import secure_filename
import json
import uuid
import markdown
import chromadb
from langchain_core.documents import Document

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.urandom(24)

# Load environment variables
load_dotenv()

api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise ValueError("GROQ_API_KEY not found in environment variables")

# Updated prompt template to encourage table format with country names
custom_prompt_template = """
Based solely on the latest news articles provided in the context, answer the user's question in detail. 
Do not use general knowledge or information outside the provided context—rely only on the news articles given.
If the context lacks relevant or recent news, state: "I don't have enough current news data to answer this question accurately."
Present your response in a table format with columns for Rank, Country, and Details, reflecting the latest news trends.

Previous conversation:
{chat_history}

Question: {question} 
Context: {context} 
Answer:
"""

ollama_model_name = "deepseek-r1:1.5b"
embeddings = OllamaEmbeddings(model="deepseek-r1:1.5b")
CHROMA_DB_PATH = "chroma_db"

pdfs_directory = 'pdfs/'
news_directory = 'news/'
UPLOAD_FOLDER = 'pdfs/'
ALLOWED_EXTENSIONS = {'pdf'}

# Create directories
os.makedirs(CHROMA_DB_PATH, exist_ok=True)
os.makedirs(news_directory, exist_ok=True)
os.makedirs(pdfs_directory, exist_ok=True)
os.makedirs('templates', exist_ok=True)

# Write template if not exists
template_path = os.path.join('templates', 'index.html')
if not os.path.exists(template_path):
    with open(template_path, 'w') as f:
        f.write("<!DOCTYPE html><!-- Your HTML template here -->")

# Global variables
last_update_time = None
db_initialized = False
update_thread = None

# Configure Flask app
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# Helper functions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def fetch_news(query, country="India", lang="en", max_results=50):  # Increased to 50 for more news
    print(f"Fetching news for query: {query}, country: {country}")
    params = {
        "api_key": "your_serpapi_key_here",  # Replace with your SerpAPI key
        "engine": "google",
        "q": query,
        "location": country,
        "google_domain": "google.co.in",
        "gl": "in",
        "hl": lang,
        "tbm": "nws",
        "cr": "countryAF",
        "num": max_results
    }
    search = GoogleSearch(params)
    results = search.get_dict()
    articles = results.get("news_results", [])
    return articles

def is_duplicate_article(new_article, existing_documents):
    new_title = new_article.get("title", "").lower()
    new_content = new_article.get("snippet", "").lower()
    
    new_signature = f"{new_title[:50]}_{new_content[:100]}"
    
    for doc in existing_documents:
        content = doc.page_content.lower()
        title_match = re.search(r"^## (.*?)$", content, re.MULTILINE)
        if title_match:
            existing_title = title_match.group(1).lower()
            content_sample = content[:200]
            existing_signature = f"{existing_title[:50]}_{content_sample[:100]}"
            
            if new_signature == existing_signature:
                return True
            
            if new_title and existing_title and (
                new_title in existing_title or 
                existing_title in new_title or
                similarity_score(new_title, existing_title) > 0.8
            ):
                return True
    
    return False

def similarity_score(text1, text2):
    words1 = set(text1.split())
    words2 = set(text2.split())
    intersection = len(words1.intersection(words2))
    union = len(words1.union(words2))
    return intersection / union if union > 0 else 0

def save_news_to_file(articles, category, country):
    if not articles:
        return None
    
    existing_documents = []
    for filename in os.listdir(news_directory):
        if filename.endswith(".txt"):
            file_path = os.path.join(news_directory, filename)
            docs = load_text_file(file_path)
            existing_documents.extend(docs)
    
    unique_articles = []
    for article in articles:
        if not is_duplicate_article(article, existing_documents):
            unique_articles.append(article)
    
    if not unique_articles:
        print(f"No unique articles found for {category} in {country}")
        return None
    
    print(f"Found {len(unique_articles)} unique articles out of {len(articles)} for {category} in {country}")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{news_directory}{category}_{country}_{timestamp}.txt"
    
    with open(filename, "w", encoding="utf-8") as f:
        f.write(f"# News about {category} for {country}\n\n")
        for article in unique_articles:
            title = article.get("title", "No Title")
            description = article.get("snippet", "No Description")
            published = article.get("date", "Unknown Date")
            source = article.get("source", "Unknown Source")
            url = article.get("link", "No URL")
            
            f.write(f"## {title}\n")
            f.write(f"Source: {source} | Published: {published}\n\n")
            f.write(f"{description}\n\n")
            f.write(f"URL: {url}\n\n")
            f.write("---\n\n")
    
    return filename

def fetch_trade_news():
    queries = {
        "Wheat_Export": '("wheat export" OR "wheat trade" OR "wheat market" OR "wheat demand" OR "wheat supply")',
        "Rice_Export": '("rice export markets" OR "rice trade countries" OR "global rice exports")',
        "Agricultural_Exports": '("agricultural exports" OR "farm exports" OR "food exports" OR "crop exports")',
        "Global_Trade": '("global trade" OR "international trade" OR "export market" OR "import market" OR "trade agreement" OR "trade policy") ',
        "Trade_Tariffs": '("trade tariffs" OR "import duties" OR "export restrictions" OR "trade barriers" OR "customs duty")',
        "Export_Markets": '("export markets" OR "foreign markets" OR "overseas buyers" OR "international customers" OR "market access") ',
        "Trade_Regulations": '("trade regulations" OR "export controls" OR "import regulations" OR "trade compliance" OR "customs procedures")',
        "Market_Trends": '("market trends" OR "consumer preferences" OR "demand patterns" OR "buying behavior" OR "consumption trends")'
    }
    
    countries = {
        "United States": "us",
        "China": "cn",
        "India": "in",
        "Japan": "jp",
        "South Korea": "kr",
        "Brazil": "br",
        "Thailand": "th",
        "Vietnam": "vn",
        "Philippines": "ph",
        "Saudi Arabia": "sa",
        "United Arab Emirates": "ae",
        "Indonesia": "id",
        "Malaysia": "my"
    }
    
    saved_files = []
    
    for category, query in queries.items():
        for country_name, country_code in countries.items():
            try:
                articles = fetch_news(query=query, country=country_name, lang="en", max_results=50)  # Increased to 50
                if articles:
                    filename = save_news_to_file(articles, category, country_code)
                    if filename:
                        saved_files.append(filename)
                time.sleep(0.5)
            except Exception as e:
                print(f"Error fetching news for {category} in {country_name}: {e}")
    
    return saved_files

def load_text_file(file_path):
    print("Loading text file from:", file_path)
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    filename = os.path.basename(file_path)
    parts = filename.split("_")
    country_code = parts[1] if len(parts) >= 3 else "unknown"
    
    articles = content.split("---\n\n")
    documents = []
    
    for article in articles:
        if not article.strip():
            continue
        
        lines = article.split("\n")
        if len(lines) < 2:
            continue
        
        title = lines[0].replace("## ", "").strip()
        source_line = lines[1].strip()
        source_match = re.search(r"Source: (.*?) \| Published: (.*)", source_line)
        source = source_match.group(1).strip() if source_match else "Unknown Source"
        published = source_match.group(2).strip() if source_match else "Unknown Date"
        
        title = title.replace(" ", "")
        published = published.split("T")[0]
        url = "No URL"
        for line in lines:
            if line.startswith("URL: "):
                url = line.replace("URL: ", "").strip()
                break
        
        content_lines = []
        in_content = False
        for line in lines[2:]:
            if line.startswith("URL: "):
                break
            if in_content or line.strip():
                in_content = True
                content_lines.append(line)
        content_text = "\n".join(content_lines).strip()
        
        doc = Document(
            page_content=content_text,
            metadata={
                "title": title,
                "source": source,
                "published": published,
                "url": url,
                "file": file_path
            }
        )
        documents.append(doc)
    
    print(f"Loaded {len(documents)} articles from {file_path}")
    return documents

def create_chunks(documents):
    print("Creating text chunks from documents...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True
    )
    text_chunks = text_splitter.split_documents(documents)
    print("Created", len(text_chunks), "text chunks.")
    return text_chunks

def get_embedding_model(ollama_model_name):
    print("Initializing embedding model:", ollama_model_name)
    try:
        embeddings = OllamaEmbeddings(model=ollama_model_name)
        print("Embedding model initialized.")
        return embeddings
    except Exception as e:
        print(f"Error initializing embedding model: {e}")
        return None

def create_vector_store(text_chunks, ollama_model_name):
    print("Creating vector store from text chunks...")
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    collection = client.get_or_create_collection(name="news_articles")
    
    embeddings_model = get_embedding_model(ollama_model_name)
    embeddings = [embeddings_model.embed_query(chunk.page_content) for chunk in text_chunks]
    
    collection.add(
        documents=[chunk.page_content for chunk in text_chunks],
        embeddings=embeddings,
        metadatas=[chunk.metadata for chunk in text_chunks],
        ids=[str(uuid.uuid4()) for _ in text_chunks]
    )
    print("Vector store created and saved locally at:", CHROMA_DB_PATH)
    return collection

def load_vector_store():
    try:
        client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        collection = client.get_collection(name="news_articles")
        print("Existing vector store loaded successfully.")
        return collection
    except Exception as e:
        print(f"Error loading vector store: {e}")
        return None

def retrieve_docs(vector_store, query):
    print("Retrieving documents from vector store for query:", query)
    embeddings_model = get_embedding_model(ollama_model_name)
    query_embedding = embeddings_model.embed_query(query)
    results = vector_store.query(
        query_embeddings=[query_embedding],
        n_results=4,  # Increased from 4
        where={"published": {"$gte": "2025-03-03"}}  # Add date filter
    )
    print("Retrieved", len(results['documents'][0]), "documents.")
    docs = [
        Document(page_content=doc, metadata=meta or {})
        for doc, meta in zip(results['documents'][0], results['metadatas'][0])
    ]
    return docs

def get_context(documents):
    print("Aggregating context from retrieved documents...")
    context_parts = []
    for doc in documents:
        title = doc.metadata.get("title", "No Title")
        published = doc.metadata.get("published", "Unknown Date")
        url = doc.metadata.get("url", "No URL")
        content = doc.page_content
        part = f"Title: {title}\nPublished: {published}\nURL: {url}\n\n{content}"
        context_parts.append(part)
    context = "\n\n---\n\n".join(context_parts)
    print("Context aggregated with metadata.")
    return context

def format_chat_history(chat_history):
    formatted_history = ""
    for message in chat_history:
        if message["role"] == "user":
            formatted_history += f"Human: {message['content']}\n"
        elif message["role"] == "assistant":
            formatted_history += f"AI: {message['content']}\n"
    return formatted_history

def clean_llm_response(response_text):
    if isinstance(response_text, str):
        content_match = re.search(r'content=[\'"](.+?)[\'"]$', response_text, re.DOTALL | re.MULTILINE)
        if content_match:
            response_text = content_match.group(1)
        content_match2 = re.search(r'content=[\'"](.*)[\'"]', response_text, re.DOTALL)
        if content_match2:
            response_text = content_match2.group(1)
        response_text = response_text.replace('\\"', '"').replace("\\'", "'")
    elif hasattr(response_text, 'content'):
        response_text = response_text.content
    
    response_text = re.sub(r'<think>.*?</think>', '', response_text, flags=re.DOTALL)
    metadata_patterns = [
        r'additional_kwargs=\{.*', 
        r'response_metadata=\{.*',
        r'id=\'run-.*',
        r'usage_metadata=\{.*'
    ]
    for pattern in metadata_patterns:
        response_text = re.sub(pattern, '', response_text, flags=re.DOTALL)
    response_text = re.sub(r'\n\s*\n', '\n\n', response_text)
    return response_text.strip()

def format_output_with_dates(documents, response):
    cleaned_response = clean_llm_response(response)
    
    # Check if the response contains a markdown table
    if '|' in cleaned_response and '-' in cleaned_response:
        # Convert markdown table to HTML
        html_table = markdown.markdown(cleaned_response, extensions=['tables'])
        return html_table
    else:
        # Format as bullet points or leave as is
        if "•" in cleaned_response or "- " in cleaned_response:
            return cleaned_response
        else:
            sections = cleaned_response.split('. ')
            bullet_points = [f"• {section.strip()}" for section in sections if section.strip()]
            return '\n'.join(bullet_points)

def answer_query(documents, query, chat_history=None):
    print("Answering query using retrieved context...")
    context = get_context(documents)
    
    formatted_history = format_chat_history(chat_history) if chat_history else ""
    
    llm_model = ChatGroq(
        temperature=0.7,
        groq_api_key=api_key,
        model="deepseek-r1-distill-llama-70b"
    )
    
    prompt = ChatPromptTemplate.from_template(custom_prompt_template)
    chain = prompt | llm_model
    response = chain.invoke({
        "question": query, 
        "context": context,
        "chat_history": formatted_history
    })
    
    response_text = response.content if hasattr(response, 'content') else str(response)
    formatted_output = format_output_with_dates(documents, response_text)
    print("Answer generated and formatted.")
    return formatted_output

def process_query(query, chat_history=None):
    vector_store = load_vector_store()
    
    if vector_store:
        print("Loaded existing vector store for query.")
        retrieved_docs = retrieve_docs(vector_store, query)
        response = answer_query(
            documents=retrieved_docs, 
            query=query,
            chat_history=chat_history
        )
        return response
    else:
        return "No data available to answer your question. Please try again later."
    
def update_news_database():
    print("Updating news database...")
    saved_files = fetch_trade_news()
    if not saved_files:
        return False
    documents = []
    for file in saved_files:
        docs = load_text_file(file)
        documents.extend(docs)
    chunks = create_chunks(documents)
    create_vector_store(chunks, ollama_model_name)
    print("News database updated successfully!")
    return True
 # Return a success indicator

# Later in the code
success = update_news_database()
if success:
    print("Database updated successfully!")
else:
    print("Failed to update database.")

def check_and_initialize_database():
    global db_initialized
    
    vector_store = load_vector_store()
    
    if vector_store:
        print("Existing vector store found.")
        db_initialized = True
        return True
    else:
        print("No existing database found or error loading it.")
        print("Initializing database with comprehensive global trade news...")
        success = update_news_database()
        if success:
            print("Database initialized with comprehensive trade news!")
            db_initialized = True
        else:
            print("Failed to initialize database.")
        return success

def check_and_update_if_needed():
    global last_update_time
    
    last_update_file = "last_update.txt"
    current_time = datetime.now()
    
    if os.path.exists(last_update_file):
        with open(last_update_file, 'r') as f:
            last_update_str = f.read().strip()
            try:
                last_update = datetime.fromisoformat(last_update_str)
                time_since_update = (current_time - last_update).total_seconds() / 3600
                if time_since_update < 1:
                    print(f"Database was updated {time_since_update:.2f} hours ago. No update needed.")
                    last_update_time = last_update
                    return False
            except ValueError:
                pass
    
    success = update_news_database()
    if success:
        with open(last_update_file, 'w') as f:
            f.write(current_time.isoformat())
    return success

def schedule_news_updates():
    while True:
        print(f"Background update: Checking for news updates at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        check_and_update_if_needed()
        time.sleep(3600)


def start_update_thread():
    global update_thread
    if update_thread is None or not update_thread.is_alive():
        update_thread = threading.Thread(target=schedule_news_updates, daemon=True)
        update_thread.start()
        print("Started background update thread")

# Initialize database and start thread
with app.app_context():
    check_and_initialize_database()
    start_update_thread()

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.json
    query = data.get('message')
    
    if 'chat_history' not in session:
        session['chat_history'] = []
    
    response = process_query(query, session['chat_history'])
    
    session['chat_history'].append({"role": "user", "content": query})
    session['chat_history'].append({"role": "assistant", "content": response})
    session.modified = True
    
    return jsonify({
        "response": response,
        "last_update": last_update_time.strftime('%Y-%m-%d %H:%M:%S') if last_update_time else None
    })

@app.route('/api/clear-chat', methods=['POST'])
def clear_chat():
    if 'chat_history' in session:
        session['chat_history'] = []
        session.modified = True
    return jsonify({"success": True})

@app.route('/api/initial-question', methods=['GET'])
def initial_question():
    if 'chat_history' not in session or not session['chat_history']:
        query = "If I want to export rice, what are the best markets and countries?"
        response = process_query(query, [])
        
        session['chat_history'] = [
            {"role": "user", "content": query},
            {"role": "assistant", "content": response}
        ]
        session.modified = True
        
        return jsonify({
            "success": True,
            "question": query,
            "response": response
        })
    return jsonify({"success": False, "message": "Chat history already exists"})

@app.route('/api/status', methods=['GET'])
def status():
    return jsonify({
        "db_initialized": db_initialized,
        "last_update": last_update_time.strftime('%Y-%m-%d %H:%M:%S') if last_update_time else None,
        "has_chat_history": 'chat_history' in session and len(session['chat_history']) > 0
    })

@app.route('/api/chat-history', methods=['GET'])
def get_chat_history():
    if 'chat_history' in session:
        return jsonify({"history": session['chat_history']})
    return jsonify({"history": []})

@app.route('/templates/index.html')
def get_index_template():
    return ""

if __name__ == '__main__':
    os.makedirs(CHROMA_DB_PATH, exist_ok=True)
    os.makedirs(news_directory, exist_ok=True)
    os.makedirs('templates', exist_ok=True)
    
    with app.app_context():
        check_and_initialize_database()
        start_update_thread()
    
    app.run(debug=True)