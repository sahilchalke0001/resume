from flask import Flask, render_template, request, jsonify,send_from_directory
import os
import io
import smtplib
import base64
import logging
from dotenv import load_dotenv
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import pdf2image
import google.generativeai as genai

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure Gemini API
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Initialize Flask App
app = Flask(__name__)

# Function to interact with Gemini API
def get_gemini_response(input_text, pdf_content, prompt):
    """
    Sends the input text, PDF content, and prompt to the Gemini API and returns the response.
    """
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content([input_text, pdf_content[0], prompt])
        return response.text
    except Exception as e:
        logger.error(f"Error interacting with Gemini API: {e}")
        return f"Error interacting with Gemini API: {e}"

# Function to process the uploaded PDF
def input_pdf_setup(uploaded_file):
    """
    Converts the uploaded PDF file into an image and encodes it in base64 format.
    """
    try:
        if uploaded_file:
            images = pdf2image.convert_from_bytes(uploaded_file.read())
            first_page = images[0]
            img_byte_arr = io.BytesIO()
            first_page.save(img_byte_arr, format='JPEG')
            img_byte_arr = img_byte_arr.getvalue()
            pdf_parts = [{
                "mime_type": "image/jpeg",
                "data": base64.b64encode(img_byte_arr).decode()
            }]
            return pdf_parts
        else:
            raise FileNotFoundError("No file uploaded")
    except Exception as e:
        logger.error(f"Error processing PDF: {e}")
        return f"Error processing PDF: {e}"

# Function to send email
def send_email(recipient, subject, body):
    """
    Sends an email using Outlook's SMTP server.
    """
    sender_email = os.getenv("EMAIL_USER")
    sender_password = os.getenv("EMAIL_PASS")
    
    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = recipient
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))
    
    try:
        # Use Outlook's SMTP server
        server = smtplib.SMTP('smtp.office365.com', 587)
        server.starttls()  # Upgrade the connection to secure
        server.login(sender_email, sender_password)
        server.sendmail(sender_email, recipient, msg.as_string())
        server.quit()
        
        logger.info("Email sent successfully!")
        return True

    except smtplib.SMTPAuthenticationError as e:
        logger.error(f"SMTP Authentication Error: {e}")
        return False
    except Exception as e:
        logger.error(f"Error sending email: {e}")
        return False

@app.route('/favicon.ico')
def favicon():
    return send_from_directory('static', 'favicon.ico', mimetype='image/vnd.microsoft.icon')

# Flask Routes
@app.route('/')
def home():
    """
    Renders the home page.
    """
    return render_template('index.html')

@app.route('/evaluate', methods=['POST'])
def evaluate():
    """
    Handles the evaluation request, processes the resume, and sends the evaluation result via email.
    """
    input_text = request.form.get('job_description')
    uploaded_file = request.files.get('resume')
    prompt_type = request.form.get('prompt_type')
    email = request.form.get('email')

    # Validate inputs
    if not input_text or not uploaded_file or not prompt_type or not email:
        return jsonify({"error": "All fields are required."}), 400

    if not uploaded_file.filename.endswith('.pdf'):
        return jsonify({"error": "Only PDF files are allowed."}), 400

    try:
        # Process the uploaded PDF
        pdf_content = input_pdf_setup(uploaded_file)

        # Define the prompt based on the prompt type
        if prompt_type == "evaluation":
            input_prompt = (
                "You are an experienced Technical Human Resource Manager, your task is to review the provided resume against the job description. "
                "Please share your professional evaluation on whether the candidate's profile aligns with the role. "
                "Highlight the strengths and weaknesses of the applicant in relation to the specified job requirements."
            )
        elif prompt_type == "percentage_match":
            input_prompt = (
                "You are a skilled ATS (Applicant Tracking System) scanner with a deep understanding of data science and ATS functionality, "
                "your task is to evaluate the resume against the provided job description. Give me the percentage of match if the resume matches "
                "the job description. First, the output should come as percentage and then keywords missing and last final thoughts."
            )
        else:
            return jsonify({"error": "Invalid prompt type."}), 400

        # Get the response from Gemini API
        response = get_gemini_response(input_text, pdf_content, input_prompt)
        
        # Prepare and send the email
        email_subject = "Your Resume Evaluation Report"
        email_body = f"{response}\n\nBest Regards,\nATS Resume Expert"
        email_sent = send_email(email, email_subject, email_body)
        
        # ✅ Return both email confirmation & result for display
        return jsonify({
            "response": response,
            "email_sent": email_sent
        }), 200

    except Exception as e:
        logger.error(f"Error in /evaluate route: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True) 