from flask import Flask, render_template, request, jsonify
import smtplib
from dotenv import load_dotenv
import base64
import os
import io
from PIL import Image
import pdf2image
import google.generativeai as genai

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

app = Flask(__name__)

# Function to interact with Gemini API
def get_gemini_response(input_text, pdf_content, prompt):
    model = genai.GenerativeModel('gemini-1.5-flash')
    response = model.generate_content([input_text, pdf_content[0], prompt])
    return response.text

# Function to process the uploaded PDF
def input_pdf_setup(uploaded_file):
    if uploaded_file:
        images = pdf2image.convert_from_bytes(uploaded_file.read())
        first_page = images[0]
        img_byte_arr = io.BytesIO()
        first_page.save(img_byte_arr, format='JPEG')
        img_byte_arr = img_byte_arr.getvalue()

        pdf_parts = [
            {
                "mime_type": "image/jpeg",
                "data": base64.b64encode(img_byte_arr).decode()
            }
        ]
        return pdf_parts
    else:
        raise FileNotFoundError("No file uploaded")

# Function to send email
def send_email(name, email, response_text):
    message = f"Subject: ATS Resume Evaluation Result\n\nHello {name},\n\nHere is your ATS evaluation result:\n\n{response_text}\n\nThank you for using ATS Resume Expert!"
    try:
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.set_debuglevel(1)  # Enable debug to see SMTP connection info
        server.starttls()
        server.login("sahilchalke0001@gmail.com", os.getenv("PASSWORD"))
        server.sendmail("sahilchalke0001@gmail.com", email, message)
        server.quit()
        print("✅ Email sent successfully!")
    except Exception as e:
        print(f"❌ Error sending email: {e}")


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/evaluate', methods=['POST'])
def evaluate():
    name = request.form.get("first_name")
    last_name = request.form.get("last_name")
    email = request.form.get("email")
    input_text = request.form.get('job_description')
    uploaded_file = request.files.get('resume')
    prompt_type = request.form.get('prompt_type')

    if not name or not last_name or not email or not input_text or not uploaded_file or not prompt_type:
        return jsonify({"error": "All fields are required."}), 400

    try:
        pdf_content = input_pdf_setup(uploaded_file)

        # Select prompt type
        if prompt_type == "evaluation":
            input_prompt = (
                "You are an experienced Technical Human Resource Manager, your task is to review the provided resume "
                "against the job description. Please share your professional evaluation on whether the candidate's "
                "profile aligns with the role. Highlight the strengths and weaknesses."
                "do not list 1,2,3 this things while giving the response"
            )
        elif prompt_type == "percentage_match":
            input_prompt = (
                "You are a skilled ATS scanner with expertise in resume analysis. Evaluate the resume against the "
                "provided job description. Give a percentage match along with missing keywords and suggestions."
                "do not list 1,2,3 this things while giving the response"
            )
        else:
            return jsonify({"error": "Invalid prompt type."}), 400

        # Get AI response
        response_text = get_gemini_response(input_text, pdf_content, input_prompt)

        # Send evaluation result via email
        send_email(name, email, response_text)

        # Render result on webpage
        return render_template('results.html', name=name, response=response_text)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
