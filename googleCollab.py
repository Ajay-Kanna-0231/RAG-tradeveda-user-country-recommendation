import os
import shutil
from dotenv import load_dotenv
from google.colab import files
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq

# Load environment variables
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError("GROQ_API_KEY not found in environment variables. Please set it using os.environ['GROQ_API_KEY'] = 'your_api_key'.")

# Define paths
FAISS_DB_PATH = "/content/drive/MyDrive/vectorstore/db_faiss"  # Save FAISS in Google Drive
PDFS_DIRECTORY = "/content/pdfs"

# Create necessary directories
os.makedirs(PDFS_DIRECTORY, exist_ok=True)
os.makedirs(os.path.dirname(FAISS_DB_PATH), exist_ok=True)

# Initialize LLM and embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

llm_model = ChatGroq(
    temperature=0.7,
    groq_api_key=api_key,
    model="deepseek-r1-distill-llama-70b"
)

custom_prompt_template = """
Use the pieces of information provided in the context to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Don't provide anything out of the given context.
Question: {question} 
Context: {context} 
Answer:
"""

# Function to upload PDFs
def upload_pdf():
    uploaded = files.upload()
    if not uploaded:
        raise ValueError("No file uploaded. Please upload a valid PDF.")
    
    file_paths = []
    for filename in uploaded.keys():
        file_path = os.path.join(PDFS_DIRECTORY, filename)
        shutil.move(filename, file_path)
        file_paths.append(file_path)
        print(f"✅ Uploaded: {filename}")
    
    return file_paths

# Load PDF
def load_pdf(file_path):
    loader = PDFPlumberLoader(file_path)
    return loader.load()

# Split document into chunks
def create_chunks(documents): 
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, add_start_index=True)
    return text_splitter.split_documents(documents)

# Get embeddings model
def get_embedding_model():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Create FAISS vector store
def create_vector_store(db_faiss_path, text_chunks):
    faiss_db = FAISS.from_documents(text_chunks, get_embedding_model())
    faiss_db.save_local(db_faiss_path)
    return faiss_db

# Retrieve similar documents
def retrieve_docs(faiss_db, query):
    return faiss_db.similarity_search(query)

# Extract context from documents
def get_context(documents):
    return "\n\n".join([doc.page_content for doc in documents])

# Generate AI response
def answer_query(documents, model, query):
    context = get_context(documents)
    prompt = ChatPromptTemplate.from_template(custom_prompt_template)
    chain = prompt | model
    response = chain.invoke({"question": query, "context": context})

   
    response_text = response.content if hasattr(response, "content") else str(response)

   
    formatted_response = " ".join(response_text.split("\n")).strip()

    return formatted_response


print("📂 **Please upload your PDF file:**")
pdf_files = upload_pdf()


if pdf_files:
    pdf_path = pdf_files[0]
    documents = load_pdf(pdf_path)
    text_chunks = create_chunks(documents)

   
    faiss_db = create_vector_store(FAISS_DB_PATH, text_chunks)

    print("\n💬 **You can now ask questions! Type 'exit' to stop.**")

    
    while True:
        query = input("\n💬 **Enter your query:** ").strip()

        if query.lower() == "exit":
            print("\n🚀 **Thank you for using AI PDF Reasoner! Have a great day.**")
            break  

        if not query:
            print("❌ Please enter a valid question!")
            continue 

        
        retrieved_docs = retrieve_docs(faiss_db, query)
        response = answer_query(retrieved_docs, llm_model, query)

       
        print("\n📌 **User Query:**", query)
        print("\n🤖 **AI Response:**", response)

else:
    print("❌ No PDF files uploaded. Please try again.")
