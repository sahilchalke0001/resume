from flask import Flask, render_template, request, jsonify
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
        # Convert PDF to image
        images = pdf2image.convert_from_bytes(uploaded_file.read())

        first_page = images[0]

        # Convert to bytes
        img_byte_arr = io.BytesIO()
        first_page.save(img_byte_arr, format='JPEG')
        img_byte_arr = img_byte_arr.getvalue()

        pdf_parts = [
            {
                "mime_type": "image/jpeg",
                "data": base64.b64encode(img_byte_arr).decode()  # Encode to base64
            }
        ]
        return pdf_parts
    else:
        raise FileNotFoundError("No file uploaded")

# Flask Routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/evaluate', methods=['POST'])
def evaluate():
    input_text = request.form.get('job_description')
    uploaded_file = request.files.get('resume')
    prompt_type = request.form.get('prompt_type')

    if not input_text or not uploaded_file or not prompt_type:
        return jsonify({"error": "All fields are required."}), 400

    try:
        pdf_content = input_pdf_setup(uploaded_file)

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

        response = get_gemini_response(input_text, pdf_content, input_prompt)
        return jsonify({"response": response}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)