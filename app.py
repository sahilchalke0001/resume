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
import datetime # Make sure you import datetime if you haven't already
from werkzeug.security import generate_password_hash # Make sure this is imported
# --- Langchain RAG Imports ---
from langchain_google_genai import GoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

# --- Load Environment Variables ---
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")
SENDER_EMAIL = os.getenv("SENDER_EMAIL")
SENDER_PASSWORD = os.getenv("PASSWORD")
MONGO_URI = os.getenv("MONGO_URI")
FLASK_SECRET_KEY = os.getenv("FLASK_SECRET_KEY")

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
try:
    client = MongoClient(MONGO_URI)
    db = client.get_database() # Auto-detects DB name from URI
    users_collection = db.users
    evaluations_collection = db.evaluations # Collection for storing results
    print("✅ MongoDB Connected Successfully.")
except Exception as e:
    print(f"❌ MongoDB Connection Error: {e}")
    client = None
    users_collection = None
    evaluations_collection = None

# --- RAG Configuration ---
KNOWLEDGE_BASE_DIR = "./knowledge_base"
VECTORSTORE_DIR = "./vector_store"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
SEARCH_K = 3

# --- RAG Setup Function ---
def initialize_rag_pipeline_lcel():
    print("--- Initializing RAG Pipeline (LCEL) ---")
    if not os.path.exists(KNOWLEDGE_BASE_DIR):
        os.makedirs(KNOWLEDGE_BASE_DIR)
        with open(os.path.join(KNOWLEDGE_BASE_DIR, "placeholder.txt"), "w") as f:
            f.write("Add knowledge base files here.")
        print(f"Created {KNOWLEDGE_BASE_DIR}. Add .txt files.")

    loader = DirectoryLoader(KNOWLEDGE_BASE_DIR, glob="**/*.txt", loader_cls=TextLoader, show_progress=True)
    documents = loader.load()

    if not documents:
        print(f"Warning: No documents in {KNOWLEDGE_BASE_DIR}. Context empty.")
        docs = []
    else:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        docs = text_splitter.split_documents(documents)
        print(f"Split into {len(docs)} chunks.")

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)

    if os.path.exists(VECTORSTORE_DIR) and os.listdir(VECTORSTORE_DIR):
        print("Loading existing vector store...")
        vector_store = Chroma(persist_directory=VECTORSTORE_DIR, embedding_function=embeddings)
    elif docs:
        print("Creating new vector store...")
        vector_store = Chroma.from_documents(docs, embeddings, persist_directory=VECTORSTORE_DIR)
        vector_store.persist()
        print(f"Vector store created at {VECTORSTORE_DIR}")
    else:
        print("No docs, no existing store. Vector store is None.")
        vector_store = None

    if not vector_store:
        print("Retriever could not be initialized.")
        return None

    retriever = vector_store.as_retriever(search_kwargs={"k": SEARCH_K})
    llm = GoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=GOOGLE_API_KEY, temperature=0.6)

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
- If "percentage_match":
  1. Calculate a percentage match (0-100%).
  2. Justify the score.
  3. List top 3-5 missing items under "Key Missing Areas:".
  4. List 3 keywords/phrases under "Keywords for Improvement:".
- If "evaluation":
  1. Provide a 2-3 paragraph evaluation (strengths, weaknesses, fit).
  2. Highlight 2-3 key skills/experiences.
  3. Conclude with a recommendation.
  4. List 3 keywords/phrases under "Keywords for Improvement:".

**Keywords for Improvement Formatting:**
- Must have the heading: "Keywords for Improvement:"
- Each keyword on a new line.
- No numbering, bullets, or special characters (only a-z, A-Z, 0-9, space).

- If Analysis Type is "evaluation":
    1. Provide a professional evaluation (2-3 paragraphs) covering strengths, weaknesses, and overall fit for the role, referencing the job description and context.
    2. Highlight 2-3 key skills or experiences that are particularly relevant or missing.
    3. Conclude with a brief recommendation (e.g., Proceed to interview, Consider for other roles, Needs significant improvement).
    4. You must output a section titled exactly: "Keywords for Improvement:".
    Under this heading, list 3 specific, actionable keywords or phrases the candidate could incorporate into their resume.
    Each keyword or phrase should be on a separate line.
    "do not list 1,2,3 or point icons this things while giving the response"
    "Only include characters a-z, A-Z, 0-9 in the response. "
    "Do not include emojis or any other special characters."


- Format this list clearly under a heading like "Key Missing Areas:".
+ You must output this list under a clear heading exactly titled "Key Missing Areas:".

- Format this list clearly under a heading like "Keywords for Improvement:".
+ You must output this list under a clear heading exactly titled "Keywords for Improvement:".

Begin Evaluation:
"""
    RAG_PROMPT = PromptTemplate.from_template(template)

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs) if docs else "No relevant context found."

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
    print("LCEL RAG Pipeline Initialized Successfully.")
    return rag_chain_lcel

rag_chain = initialize_rag_pipeline_lcel()

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
        print("❌ Email config missing. Skipping.")
        return False
    message = f"Subject: {subject}\n\nHello {name},\n\n{body}\n\nThank you,\nATS Resume Expert"
    try:
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(SENDER_EMAIL, SENDER_PASSWORD)
        server.sendmail(SENDER_EMAIL, recipient_email, message.encode('utf-8'))
        server.quit()
        print(f"✅ Email sent to {recipient_email}!")
        return True
    except Exception as e:
        print(f"❌ Error sending email: {e}")
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

        # --- IMPORTANT: Clean inputs here, just like in login ---
        email = request.form.get('email').strip().lower() # Clean email
        password = request.form.get('password').strip()   # Clean password
        # -------------------------------------------------------

        first_name = request.form.get('first_name')
        last_name = request.form.get('last_name')

        if not all([email, password, first_name, last_name]):
            flash('All fields are required.', 'danger')
            return render_template('register.html')

        # Now, this check will use the cleaned email:
        if users_collection.find_one({'email': email}):
            flash('Email already registered. Please log in.', 'warning')
            return redirect(url_for('login'))

        # The password hashing is correct, but we're hashing the cleaned password now
        hashed_password = generate_password_hash(password, method='pbkdf2:sha256')
        users_collection.insert_one({
            'first_name': first_name, 'last_name': last_name,
            'email': email, # Storing the cleaned email
            'password': hashed_password,
            'created_at': datetime.datetime.utcnow()
        })
        flash('Registration successful! Please log in.', 'success')
        return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        # --- Debugging (remove for production) ---
        print("Login POST request received.")
        
        if users_collection is None:
            flash('Database connection or collection initialization failed.', 'danger')
            # --- Debugging (remove for production) ---
            print("Error: users_collection is None.")
            # ------------------------------------------
            return render_template('login.html')

        # Get and clean input
        email = request.form.get('email')
        password = request.form.get('password')

        # --- Debugging (remove or be careful with password in production) ---
        print(f"Email entered: '{email}'")
        print(f"Password entered (raw): '{password}'")
        # ------------------------------------------------------------------

        # Basic input validation
        if not email or not password:
            flash('Email and password are required.', 'danger')
            # --- Debugging (remove for production) ---
            print("Error: Email or password not provided.")
            # ------------------------------------------
            return render_template('login.html')

        # Clean inputs (strip whitespace and lowercase email for consistency)
        email = email.strip().lower()
        password = password.strip()

        # --- Debugging (remove for production) ---
        print(f"Cleaned Email: '{email}'")
        print(f"Cleaned Password: '{password}'")
        # ------------------------------------------

        # Find the user by email
        user = users_collection.find_one({'email': email})

        # --- Debugging (remove for production) ---
        if user:
            print(f"User found in DB for email '{email}'.")
            print(f"Stored hashed password for user: '{user.get('password')}'") # Use .get to prevent KeyError if 'password' isn't present
        else:
            print(f"No user found in DB for email '{email}'.")
        # ------------------------------------------

        # Check if user exists and password matches
        if user and check_password_hash(user.get('password', ''), password): # Use .get for password for safety
            session['user_id'] = str(user['_id'])
            session['user_email'] = user['email']
            session['user_name'] = user.get('first_name', 'Guest') # Use .get for other fields too, in case they're missing
            flash('Logged in successfully!', 'success')
            # --- Debugging (remove for production) ---
            print("Login successful. Redirecting to app_home.")
            # ------------------------------------------
            return redirect(url_for('app_home')) # Redirect to the main app page
        else:
            flash('Invalid email or password.', 'danger')
            # --- Debugging (remove for production) ---
            print("Login failed: Invalid email or password (user not found or password mismatch).")
            # ------------------------------------------
            return render_template('login.html')

    # For GET requests, just render the login page
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
        # --- Extract Resume Text (Unchanged) ---
        print("Extracting text from resume PDF...")
        resume_text = extract_text_from_pdf(uploaded_file)
        print(f"Extracted {len(resume_text)} characters from resume.")

        # --- Perform RAG Analysis using LCEL ---
        print(f"Performing RAG analysis (Type: {prompt_type}) using LCEL...")
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
                # Normalize line for section detection
                clean_line = line.lower().strip().replace("**", "").replace(":", "")
                
                # Detect keyword section start
                if not keyword_section_found and SECTION_START in clean_line:
                    keyword_section_found = True
                    continue
                    
                if keyword_section_found:
                    # Check for section end conditions
                    stripped = line.strip()
                    if any(clean_line.startswith(patt) for patt in SECTION_END_PATTERNS):
                        break
                    
                    # Process non-empty lines in section
                    if stripped:
                        # Clean and split keywords
                        cleaned = re.sub(r'^\W+', '', stripped)  # Remove leading symbols
                        if ',' in cleaned:
                            temp_keywords.extend([kw.strip() for kw in cleaned.split(',')])
                        else:
                            temp_keywords.append(cleaned)

            # Filter and prioritize keywords
            keywords_for_improvement = sorted(
                {kw for kw in temp_keywords if 3 < len(kw) < 50},  # Validate length
                key=lambda x: (-len(x), x),  # Sort by length then alphabetically
            )[:5]

        except Exception as parse_err:
            print(f"Keyword parsing error: {parse_err}")
            keywords_for_improvement = []

        # Fallback mechanism
        if not keywords_for_improvement:
            print("Using fallback keywords")
            keywords_for_improvement = [
                "resume writing tips", 
                "job interview skills", 
                "effective communication"
            ]
   
        if YOUTUBE_API_KEY:
            youtube_links = get_youtube_links(keywords_for_improvement[:3], YOUTUBE_API_KEY)  # Use the top 3 keywords
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
            print("✅ Evaluation record saved.")
        except Exception as db_err:
            print(f"❌ DB Save Error: {db_err}")
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
        print(f"❌ Evaluation Error: {e}")
        traceback.print_exc()
        return render_template('error.html', error_message="An internal server error occurred."), 500

# --- Main Execution Block ---
if __name__ == '__main__':
    for dir_path in [KNOWLEDGE_BASE_DIR, VECTORSTORE_DIR]:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
    app.run(host="0.0.0.0", port=7860, debug=True) # debug=True is helpful for development