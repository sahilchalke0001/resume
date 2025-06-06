import os
import re
import io
import smtplib
from dotenv import load_dotenv
from flask import Flask, render_template, request, jsonify, redirect, url_for, flash, session
import fitz  # PyMuPDF
import google.generativeai as genai
import traceback
from pymongo import MongoClient
from werkzeug.security import generate_password_hash, check_password_hash
from functools import wraps
from bson.objectid import ObjectId
import datetime
from courses import get_youtube_links

# --- Langchain RAG Imports ---
from langchain_google_genai import GoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

# NEW: Import Mongo Atlas Vector Search
from langchain_mongodb import MongoDBAtlasVectorSearch
from pymongo import MongoClient

# --- Load Environment Variables ---
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")
SENDER_EMAIL = os.getenv("SENDER_EMAIL")
SENDER_PASSWORD = os.getenv("PASSWORD")
MONGO_URI = os.getenv("MONGO_URI")
FLASK_SECRET_KEY = os.getenv("FLASK_SECRET_KEY")
MONGO_DB_NAME = os.getenv("MONGO_DB_NAME") 
MONGO_KB_COLLECTION = os.getenv("MONGO_KB_COLLECTION") 
MONGO_VECTOR_INDEX = os.getenv("MONGO_VECTOR_INDEX") 

# --- Environment Variable Checks ---
if not GOOGLE_API_KEY: raise ValueError("GOOGLE_API_KEY not found.")
if not MONGO_URI: raise ValueError("MONGO_URI not found.")
if not FLASK_SECRET_KEY: raise ValueError("FLASK_SECRET_KEY not found.")
if not SENDER_PASSWORD: print("Warning: SENDER_PASSWORD not set. Email may fail.")
if not SENDER_EMAIL: print("Warning: SENDER_EMAIL not set. Email may fail.")
if not YOUTUBE_API_KEY: print("Warning: YOUTUBE_API_KEY not set. Youtube will fail.")

# --- Configure Google AI ---
genai.configure(api_key=GOOGLE_API_KEY)

# --- Flask App Initialization ---
app = Flask(__name__)
app.secret_key = FLASK_SECRET_KEY

# --- MongoDB Setup ---
client = None # Initialize client to None
try:
    client = MongoClient(MONGO_URI)
    db = client.get_database(MONGO_DB_NAME) # Use specified DB name
    users_collection = db.users
    evaluations_collection = db.evaluations # Collection for storing results
    knowledge_chunks_collection = db[MONGO_KB_COLLECTION] # NEW: Collection for KB chunks
    print(" MongoDB Connected Successfully.")
except Exception as e:
    print(f" MongoDB Connection Error: {e}")
    client = None
    users_collection = None
    evaluations_collection = None
    knowledge_chunks_collection = None

# --- RAG Configuration (Constants) ---
KNOWLEDGE_BASE_DIR = "./knowledge_base" # Still used for initial loading
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
SEARCH_K = 3

# --- RAG Setup Function (UPDATED) ---
def initialize_rag_pipeline_lcel():
    print("--- Initializing RAG Pipeline (LCEL) with MongoDB Atlas Vector Search ---")

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)
    llm = GoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=GOOGLE_API_KEY, temperature=0.6)

    # NEW: Connect to MongoDB Atlas Vector Search
    if knowledge_chunks_collection is None:
        print("Error: knowledge_chunks_collection is not initialized. Cannot set up Atlas Vector Search.")
        return None

    try:
        vector_store = MongoDBAtlasVectorSearch(
            collection=knowledge_chunks_collection,
            embedding=embeddings,
            index_name=MONGO_VECTOR_INDEX, # Name of your Atlas Search index
            text_key="content",           # Field in your MongoDB document containing the text
            embedding_key="embedding"     # Field in your MongoDB document containing the vector
        )
        retriever = vector_store.as_retriever(search_kwargs={"k": SEARCH_K})
        print("✅ MongoDB Atlas Vector Search Retriever Initialized.")
    except Exception as e:
        print(f"❌ Error initializing MongoDB Atlas Vector Search: {e}")
        print("Falling back to no context for RAG pipeline.")
        retriever = None # Fallback if vector search fails

    template = """
    You are an expert ATS (Applicant Tracking System) evaluator and career advisor AI.
    Analyze the provided Resume against the Job Description, using the Context for guidance.

    Context:
    {context}

    Job Description:
    {job_description}

    Resume Text:
    {resume_text}

    Analysis Type Requested: {analysis_type}

    Instructions:
    - If Analysis Type is "percentage_match":
    1. Calculate a percentage match (0–100) based on how well the resume aligns with the job description and context.
    2. Justify the score clearly.
    3. Under the heading **Key Missing Areas:** list the top 3–5 missing skills, qualifications, or experiences.
    4. Under the heading **Keywords for Improvement:** list 3 keywords or phrases that the candidate should consider including in their resume.
        - Each keyword must be on a separate line.
        - Do not use bullets, numbers, or special characters.
        - Only include characters a–z, A–Z, 0–9, and spaces.
        - Do not include "**" in the RESULT.

    - If Analysis Type is "evaluation":
    1. Write a professional evaluation (2–3 paragraphs) summarizing strengths, weaknesses, and overall fit, referencing the job description and context.
    2. Highlight 2–3 key relevant or missing skills/experiences.
    3. Conclude with a brief recommendation such as:
        - Proceed to interview
        - Consider for other roles
        - Needs significant improvement
    4. Under the heading **Keywords for Improvement:** list 3 specific, actionable keywords or phrases the candidate could add to their resume.
        - Each keyword must be on a new line.
        - Do not include numbers, bullets, or any special characters.
        - Only include characters a–z, A–Z, 0-9, and spaces.
        - Do not include "**" in the RESULT

    - Output must include the following **two exact headings** (spelled and capitalized as shown):
    - Key Missing Areas:
    - Keywords for Improvement:

    - If the job description and resume are in a language other than English, the analysis output (percentage match or evaluation) must be in that same language.

    Begin Evaluation:
    """

    RAG_PROMPT = PromptTemplate.from_template(template)

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs) if docs else "No relevant context found."

    # Modified RAG Chain to conditionally use retriever
    if retriever:
        rag_chain_lcel = (
            {
                "context": (lambda x: x["job_description"]) | retriever | format_docs,
                "job_description": (lambda x: x["job_description"]),
                "resume_text": (lambda x: x["resume_text"]),
                "analysis_type": (lambda x: x["analysis_type"]),
            }
            | RAG_PROMPT
            | llm
            | StrOutputParser()
        )
        print("LCEL RAG Pipeline Initialized with Atlas Vector Search.")
    else:
        # Fallback chain if retriever couldn't be initialized
        rag_chain_lcel = (
            {
                "context": RunnableLambda(lambda x: "No external knowledge base available."), # Provide a static message
                "job_description": (lambda x: x["job_description"]),
                "resume_text": (lambda x: x["resume_text"]),
                "analysis_type": (lambda x: x["analysis_type"]),
            }
            | RAG_PROMPT
            | llm
            | StrOutputParser()
        )
        print("LCEL RAG Pipeline Initialized without Atlas Vector Search (fallback).")

    return rag_chain_lcel

# Initialize the RAG pipeline
rag_chain = initialize_rag_pipeline_lcel()

# --- Initial Data Loading Function (NEW) ---
# This function will load your local knowledge base files into MongoDB.
# You should run this ONCE when you deploy or when your knowledge base changes.
def load_knowledge_base_to_mongodb():
    print("--- Loading Knowledge Base to MongoDB ---")
    if knowledge_chunks_collection is None:
        print("Error: knowledge_chunks_collection is not initialized. Cannot load data.")
        return

    # Clear existing data if you want to re-ingest
    # knowledge_chunks_collection.delete_many({})
    # print("Cleared existing knowledge base documents in MongoDB.")

    loader = DirectoryLoader(KNOWLEDGE_BASE_DIR, glob="**/*.txt", loader_cls=TextLoader, show_progress=True)
    documents = loader.load()

    if not documents:
        print(f"Warning: No documents in {KNOWLEDGE_BASE_DIR}. No data to load to MongoDB.")
        return

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    docs = text_splitter.split_documents(documents)
    print(f"Split local documents into {len(docs)} chunks for MongoDB.")

    # Generate embeddings and prepare for insertion
    embeddings_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)
    documents_to_insert = []
    for i, doc in enumerate(docs):
        try:
            # Generate embedding for each chunk
            embedding = embeddings_model.embed_query(doc.page_content)
            documents_to_insert.append({
                "content": doc.page_content,
                "embedding": embedding,
                "metadata": doc.metadata, # Keep any metadata from source files
                "chunk_id": i,
                "created_at": datetime.datetime.utcnow()
            })
        except Exception as e:
            print(f"Error embedding chunk {i}: {e}")
            continue

    if documents_to_insert:
        try:
            knowledge_chunks_collection.insert_many(documents_to_insert)
            print(f" Successfully inserted {len(documents_to_insert)} chunks into MongoDB collection '{MONGO_KB_COLLECTION}'.")
        except Exception as e:
            print(f" Error inserting chunks into MongoDB: {e}")
    else:
        print("No documents prepared for insertion after embedding.")

# --- PDF Text Extraction ---
def extract_text_from_pdf(uploaded_file):
    if not uploaded_file: raise FileNotFoundError("No file uploaded")
    try:
        uploaded_file.seek(0) # IMPORTANT: Go to start of file stream
        pdf_data = uploaded_file.read()
        doc = fitz.open(stream=pdf_data, filetype="pdf")
        text = "".join(page.get_text() for page in doc)
        doc.close()
        if not text.strip(): raise ValueError("Could not extract text from PDF.")
        return text
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
        raise ValueError(f"Failed to process PDF: {e}")

# --- Send Email ---
def send_email(name, recipient_email, subject, body):
    if not SENDER_EMAIL or not SENDER_PASSWORD:
        print(" Email config missing. Skipping.")
        return False
    message = f"Subject: {subject}\n\nHello {name},\n\n{body}\n\nThank you,\nATS Resume Expert"
    try:
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(SENDER_EMAIL, SENDER_PASSWORD)
        server.sendmail(SENDER_EMAIL, recipient_email, message.encode('utf-8'))
        server.quit()
        print(f" Email sent to {recipient_email}!")
        return True
    except Exception as e:
        print(f" Error sending email: {e}")
        return False

# --- Login Required Decorator ---
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash('Please log in to access this page.', 'warning')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

# --- Authentication Routes ---
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        if users_collection is None:
            flash('Database connection failed.', 'danger')
            return render_template('register.html')

        email = request.form.get('email').strip().lower()
        password = request.form.get('password').strip()
        first_name = request.form.get('first_name')
        last_name = request.form.get('last_name')

        if not all([email, password, first_name, last_name]):
            flash('All fields are required.', 'danger')
            return render_template('register.html')

        if users_collection.find_one({'email': email}):
            flash('Email already registered. Please log in.', 'warning')
            return redirect(url_for('login'))

        password_pattern = r'^(?=.*\d)(?=.*[!@#$%^&*()_+{}\[\]:;<>,.?~\\-]).{7,}$'

        if not re.match(password_pattern, password):
            flash('Password must be at least 7 characters long and include at least one digit and one special character.', 'danger')
            return render_template('register.html',
                email=email,
                first_name=first_name,
                last_name=last_name
        )

        hashed_password = generate_password_hash(password, method='pbkdf2:sha256')
        users_collection.insert_one({
            'first_name': first_name, 'last_name': last_name,
            'email': email,
            'password': hashed_password,
            'created_at': datetime.datetime.utcnow()
        })
        flash('Registration successful! Please log in.', 'success')
        return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        print("Login POST request received.")

        if users_collection is None:
            flash('Database connection or collection initialization failed.', 'danger')
            print("Error: users_collection is None.")
            return render_template('login.html')

        email = request.form.get('email')
        password = request.form.get('password')

        print(f"Email entered: '{email}'")
        print(f"Password entered (raw): '{password}'")

        if not email or not password:
            flash('Email and password are required.', 'danger')
            print("Error: Email or password not provided.")
            return render_template('login.html')

        email = email.strip().lower()
        password = password.strip()

        print(f"Cleaned Email: '{email}'")
        print(f"Cleaned Password: '{password}'")

        user = users_collection.find_one({'email': email})

        if user:
            print(f"User found in DB for email '{email}'.")
            print(f"Stored hashed password for user: '{user.get('password')}'")
        else:
            print(f"No user found in DB for email '{email}'.")

        if user and check_password_hash(user.get('password', ''), password):
            session['user_id'] = str(user['_id'])
            session['user_email'] = user['email']
            session['user_name'] = user.get('first_name', 'Guest')
            flash('Logged in successfully!', 'success')
            print("Login successful. Redirecting to app_home.")
            return redirect(url_for('app_home'))
        else:
            flash('Invalid email or password.', 'danger')
            print("Login failed: Invalid email or password (user not found or password mismatch).")
            return render_template('login.html')

    return render_template('login.html')


@app.route('/logout')
def logout():
    session.clear()
    flash('You have been logged out.', 'info')
    return redirect(url_for('login'))

# --- Main Application Routes ---
@app.route('/')
def root():
    if 'user_id' in session:
        return redirect(url_for('app_home'))
    return redirect(url_for('login'))

@app.route('/app')
@login_required
def app_home():
    user_name = session.get('user_name', 'Guest')
    return render_template('index.html', user_name=user_name)

@app.route('/evaluate', methods=['POST'])
@login_required
def evaluate():
    if not rag_chain:
        flash("Analysis engine not ready. Please try again later.", "danger")
        return redirect(url_for('app_home'))
    if evaluations_collection is None:
        flash("Database not ready. Please try again later.", "danger")
        return redirect(url_for('app_home'))

    job_description = request.form.get('job_description', "").strip()
    uploaded_file = request.files.get('resume')
    prompt_type = request.form.get('prompt_type')
    user_id = session.get('user_id')

    if not all([job_description, uploaded_file, prompt_type]):
        flash("Missing job description, resume, or analysis type.", "danger")
        return redirect(url_for('app_home'))

    if prompt_type not in ["evaluation", "percentage_match"]:
        return jsonify({"error": "Invalid analysis type selected."}), 400

    try:
        print("Extracting text from resume PDF...")
        resume_text = extract_text_from_pdf(uploaded_file)
        print(f"Extracted {len(resume_text)} characters from resume.")

        print(f"Performing RAG analysis (Type: {prompt_type}) using LCEL with Atlas Vector Search...")
        lcel_input = {
            "job_description": job_description,
            "resume_text": resume_text,
            "analysis_type": prompt_type
        }
        analysis_text = rag_chain.invoke(lcel_input)
        print("RAG analysis complete.")

        keywords_for_improvement = []
        youtube_links = []

        try:
            lines = analysis_text.splitlines()
            keyword_section_found = False
            temp_keywords = []

            # Define section boundaries
            SECTION_START = "keywords for improvement"
            SECTION_END_PATTERNS = (
                "key missing areas",
                "key areas",
                "justification",
                "analysis type:"
            )

            for line in lines:
                clean_line = line.lower().strip().replace("**", "").replace(":", "")

                if not keyword_section_found and SECTION_START in clean_line:
                    keyword_section_found = True
                    continue

                if keyword_section_found:
                    stripped = line.strip()
                    if any(clean_line.startswith(patt) for patt in SECTION_END_PATTERNS):
                        break

                    if stripped:
                        cleaned = re.sub(r'^\W+', '', stripped)
                        if ',' in cleaned:
                            temp_keywords.extend([kw.strip() for kw in cleaned.split(',')])
                        else:
                            temp_keywords.append(cleaned)

            keywords_for_improvement = sorted(
                {kw for kw in temp_keywords if 3 < len(kw) < 50},
                key=lambda x: (-len(x), x),
            )[:5]

        except Exception as parse_err:
            print(f"Keyword parsing error: {parse_err}")
            keywords_for_improvement = []

        if not keywords_for_improvement:
            print("Using fallback keywords")
            keywords_for_improvement = [
                "resume writing tips",
                "job interview skills",
                "effective communication"
            ]

        if YOUTUBE_API_KEY:
            youtube_links = get_youtube_links(keywords_for_improvement[:3], YOUTUBE_API_KEY)
        else:
            print("YouTube API Key not configured. Skipping video search.")

        # Save to MongoDB
        try:
            evaluation_record = {
                "user_id": ObjectId(user_id),
                "job_description": job_description,
                "resume_text": resume_text,
                "analysis_type": prompt_type,
                "analysis_result": analysis_text,
                "keywords_for_improvement": keywords_for_improvement,
                "youtube_links": youtube_links,
                "created_at": datetime.datetime.utcnow()
            }
            evaluations_collection.insert_one(evaluation_record)
            print(" Evaluation record saved.")
        except Exception as db_err:
            print(f" DB Save Error: {db_err}")
            flash("Could not save your results, but analysis is complete.", "warning")

        # Send Email
        user_name = session.get('user_name')
        user_email = session.get('user_email')
        email_subject = f"Your ATS Resume {prompt_type.replace('_', ' ').title()}"
        email_body = f"Here is your analysis:\n\n{analysis_text}"
        if youtube_links:
            email_body += "\n\nSuggested videos:\n" + "\n".join([f"- {v['title']}: {v['url']}" for v in youtube_links])
        send_email(user_name, user_email, email_subject, email_body)

        # Render Results
        return render_template(
            'results.html',
            name=user_name,
            response=analysis_text,
            youtube_links=youtube_links,
            youtube_queries=keywords_for_improvement
        )

    except (FileNotFoundError, ValueError) as e:
        flash(str(e), "danger")
        return redirect(url_for('app_home'))
    except Exception as e:
        print(f" Evaluation Error: {e}")
        traceback.print_exc()
        return render_template('error.html', error_message="An internal server error occurred."), 500

# --- Main Execution Block ---
if __name__ == '__main__':
    # You still need your local knowledge_base directory to initially load data
    if not os.path.exists(KNOWLEDGE_BASE_DIR):
        os.makedirs(KNOWLEDGE_BASE_DIR)
        with open(os.path.join(KNOWLEDGE_BASE_DIR, "placeholder.txt"), "w") as f:
            f.write("Add your knowledge base .txt files here to be loaded into MongoDB.")


    if client is not None and knowledge_chunks_collection is not None: # Ensure DB connection is established
        # Check if the collection is empty before loading, to avoid duplicates on every restart
        if knowledge_chunks_collection.count_documents({}) == 0:
            load_knowledge_base_to_mongodb()
        else:
            print("Knowledge base collection not empty, skipping initial load.")


    app.run(host="0.0.0.0", port=7860, debug=True)