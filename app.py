import os
import re
import io
import base64
import smtplib
from dotenv import load_dotenv
from flask import Flask, render_template, request, jsonify
import fitz # PyMuPDF for PDF text extraction
import google.generativeai as genai
import traceback # For detailed error logging
# Import the function from your courses.py file
from courses import get_youtube_links # Assuming your file is named courses.py
# --- Langchain RAG Imports ---
from langchain_google_genai import GoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, TextLoader
# from langchain.chains import RetrievalQA # No longer using RetrievalQA
from langchain.prompts import PromptTemplate
# --- LCEL Imports ---
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

# --- Load Environment Variables ---
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
# Load your YouTube API Key *before* checking it
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")
SENDER_EMAIL = os.getenv("SENDER_EMAIL") # Corrected placeholder email format
SENDER_PASSWORD = os.getenv("PASSWORD") # Your email app password or token

if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in environment variables.")
if not SENDER_PASSWORD:
    print("Warning: SENDER_PASSWORD environment variable not set. Email sending will likely fail.")
# Use a more standard placeholder or an empty string if no default
if not SENDER_EMAIL:
    print("Warning: SENDER_EMAIL environment variable not set or default. Using placeholder.")

# Now check YOUTUBE_API_KEY after it's loaded
if not YOUTUBE_API_KEY:
    print("Warning: YOUTUBE_API_KEY environment variable not set. YouTube video search will not work.")


genai.configure(api_key=GOOGLE_API_KEY)

# --- Flask App Initialization ---
app = Flask(__name__)

# --- RAG Configuration ---
KNOWLEDGE_BASE_DIR = "./knowledge_base" # Directory containing your text files/documents
VECTORSTORE_DIR = "./vector_store"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
SEARCH_K = 3

# --- RAG Setup Function using LCEL ---
def initialize_rag_pipeline_lcel():
    print("--- Initializing RAG Pipeline (LCEL) ---")
    print("Loading knowledge base documents...")

    if not os.path.exists(KNOWLEDGE_BASE_DIR):
        os.makedirs(KNOWLEDGE_BASE_DIR)
        with open(os.path.join(KNOWLEDGE_BASE_DIR, "placeholder.txt"), "w") as f:
            f.write("Add your knowledge base files here (e.g., job requirements, skills examples, resume best practices).")
        print(f"Created knowledge base directory: {KNOWLEDGE_BASE_DIR}")
        print("Please add relevant .txt files to this directory.")

    loader = DirectoryLoader(KNOWLEDGE_BASE_DIR, glob="**/*.txt", loader_cls=TextLoader, show_progress=True)
    documents = loader.load()

    if not documents:
        print(f"Warning: No documents found in {KNOWLEDGE_BASE_DIR}. RAG context will be empty.")
        docs = []
    else:
        print(f"Splitting {len(documents)} documents...")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        docs = text_splitter.split_documents(documents)
        print(f"Split into {len(docs)} chunks.")

    print("Initializing embedding model...")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)

    print("Initializing vector store...")
    if os.path.exists(VECTORSTORE_DIR) and os.listdir(VECTORSTORE_DIR):
        print("Loading existing vector store...")
        vector_store = Chroma(persist_directory=VECTORSTORE_DIR, embedding_function=embeddings)
    elif docs:
        print("Creating new vector store...")
        vector_store = Chroma.from_documents(docs, embeddings, persist_directory=VECTORSTORE_DIR)
        vector_store.persist()
        print(f"Vector store created and persisted at {VECTORSTORE_DIR}")
    else:
        print("No documents to create vector store from, and no existing store found.")
        vector_store = None

    if vector_store:
        retriever = vector_store.as_retriever(search_kwargs={"k": SEARCH_K})
        print(f"Retriever initialized to fetch top {SEARCH_K} chunks.")
    else:
        retriever = None
        print("Retriever could not be initialized. RAG will not function correctly.")
        return None

    print("Initializing LLM...")
    llm = GoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=GOOGLE_API_KEY, temperature=0.6)

    template = """
You are an expert ATS (Applicant Tracking System) evaluator and career advisor AI.
Analyze the provided Resume against the Job Description, considering the following Context from our knowledge base on skills and best practices.

Context:
{context}

Job Description:
{job_description}

Resume Text:
{resume_text}

Based on all the above information, perform the requested analysis:

Analysis Type Requested: {analysis_type}

Instructions:
- If Analysis Type is "percentage_match":
    1. Calculate a percentage match score (0-100%).
    2. Provide a brief justification for the score.
    3. List the top 3-5 most important missing keywords or qualifications from the resume compared to the job description. Format this list clearly under a heading like "Key Missing Areas:".
    4. You must output a section titled exactly: "Keywords for Improvement:".
    Under this heading, list 3 specific, actionable keywords or phrases the candidate could incorporate into their resume.
    Each keyword or phrase should be on a separate line.
    "do not list 1,2,3 or point icons this things while giving the response"
    "Only include characters a-z, A-Z, 0-9 in the response. "
    "Do not include emojis or any other special characters."
    "Do not include _ also"


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

    RAG_PROMPT = PromptTemplate(
        template=template,
        input_variables=["context", "job_description", "resume_text", "analysis_type"]
    )

    print("Constructing LCEL RAG chain...")

    def format_docs(docs):
        """Helper function to format retrieved documents."""
        if not docs:
            return "No relevant context found in the knowledge base."
        return "\n\n".join(doc.page_content for doc in docs)

    def get_retriever_input(input_dict):
        return input_dict["job_description"]

    rag_chain_lcel = (
        {
            "context": RunnableLambda(get_retriever_input) | retriever | RunnableLambda(format_docs),
            "job_description": RunnablePassthrough(),
            "resume_text": RunnablePassthrough(),
            "analysis_type": RunnablePassthrough()
        }
        | RAG_PROMPT
        | llm
        | StrOutputParser()
    )

    print("LCEL RAG Pipeline Initialized Successfully.")
    return rag_chain_lcel

# --- Initialize RAG Pipeline Globally ---
rag_chain = initialize_rag_pipeline_lcel()

# --- PDF Text Extraction Function (Unchanged) ---
def extract_text_from_pdf(uploaded_file):
    """Extracts text content from an uploaded PDF file."""
    if not uploaded_file:
        raise FileNotFoundError("No file uploaded")
    try:
        pdf_data = uploaded_file.read()
        doc = fitz.open(stream=pdf_data, filetype="pdf")
        text = "".join(page.get_text() for page in doc)
        doc.close()
        if not text.strip():
            raise ValueError("Could not extract text from PDF. Is it image-based or corrupted?")
        return text
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
        raise ValueError(f"Failed to process PDF: {e}")

# --- Function to Send Email (Unchanged) ---
def send_email(name, recipient_email, subject, body):
    """Sends an email using SMTP."""
    if not SENDER_EMAIL or not SENDER_PASSWORD:
        print("❌ Email configuration missing (SENDER_EMAIL or SENDER_PASSWORD). Skipping email.")
        return False
    message = f"Subject: {subject}\n\nHello {name},\n\n{body}\n\nThank you,\nATS Resume Expert"
    try:
        message_bytes = message.encode('utf-8')
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(SENDER_EMAIL, SENDER_PASSWORD)
        server.sendmail(SENDER_EMAIL, recipient_email, message_bytes)
        server.quit()
        print(f"✅ Email sent successfully to {recipient_email}!")
        return True
    except smtplib.SMTPAuthenticationError:
        print(f"❌ SMTP Authentication Error: Check SENDER_EMAIL and SENDER_PASSWORD/.env file.")
        print("   Ensure 'less secure app access' is enabled or use an App Password if using Gmail 2FA.")
        return False
    except Exception as e:
        print(f"❌ Error sending email to {recipient_email}: {e}")
        return False


# --- Flask Routes ---
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/evaluate', methods=['POST'])
def evaluate():
    """Handles form submission, performs RAG analysis, searches YouTube, and renders results."""
    if not rag_chain:
        return jsonify({"error": "RAG pipeline not initialized. Check server logs."}), 500

    # --- Get Form Data (Unchanged) ---
    name = request.form.get("first_name", "").strip()
    last_name = request.form.get("last_name", "").strip()
    email = request.form.get("email", "").strip()
    job_description = request.form.get('job_description', "").strip()
    uploaded_file = request.files.get('resume')
    prompt_type = request.form.get('prompt_type')

    # --- Basic Validation (Unchanged) ---
    if not all([name, last_name, email, job_description, uploaded_file, prompt_type]):
        return jsonify({"error": "Missing required fields. Please fill out the entire form."}), 400
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

# ... (rest of the code remains the same)

        

   
        if YOUTUBE_API_KEY:
            youtube_links = get_youtube_links(keywords_for_improvement[:3], YOUTUBE_API_KEY)  # Use the top 3 keywords
        else:
            print("YouTube API Key not configured. Skipping video search.")

        # --- Prepare Email Content (Optional) ---
        email_subject = f"Your ATS Resume {prompt_type.replace('_', ' ').title()}"
        email_body = f"Here is your resume analysis based on the job description provided:\n\n{analysis_text}"
        if youtube_links: # Check if the list is not empty
            email_body += "\n\nSuggested videos to help you improve:\n"
            # Create a list of formatted strings (e.g., "Title: URL") from the dictionaries
            video_lines_for_email = [f"- {video['title']}: {video['url']}" for video in youtube_links]
            # Join these formatted strings with newlines
            email_body += "\n".join(video_lines_for_email)
# --- End of corrected section ---

        email_body += "\n\nBest of luck with your job search!\nATS Resume Expert" # Or your closing


        # --- Send Email (Optional) ---
        send_email(f"{name} {last_name}", email, email_subject, email_body)

        # --- Render results.html template directly ---
        full_name = f"{name} {last_name}"
        return render_template(
            'results.html',
            name=full_name,
            response=analysis_text,
            youtube_links=youtube_links
        )

    except FileNotFoundError as e:
        print(f"Error: {e}")
        return render_template('error.html', error_message=str(e)), 400
    except ValueError as e:
        print(f"Error: {e}")
        return render_template('error.html', error_message=str(e)), 400
    except Exception as e:
        print(f"An unexpected error occurred during evaluation: {e}")
        traceback.print_exc()
        return render_template('error.html', error_message="An internal server error occurred during analysis. Please try again later."), 500


# --- Main Execution Block (Unchanged) ---
if __name__ == '__main__':
    if not os.path.exists(KNOWLEDGE_BASE_DIR):
        os.makedirs(KNOWLEDGE_BASE_DIR)
    if not os.path.exists(VECTORSTORE_DIR):
        os.makedirs(VECTORSTORE_DIR)
    app.run(host="0.0.0.0", port=7860)