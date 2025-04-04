# ATS Resume Tracker

## Overview
The **ATS Resume Tracker** is a web-based application built using **Python Flask** and **Gemini API** to evaluate resumes against job descriptions. It provides a **percentage match** between a resume and job description and **summarizes the resume**, highlighting the candidate's **strengths and weaknesses**. The results are displayed on the website and emailed to the user using **SMTP**.

## Features
- **Resume Parsing**: Extracts relevant information from the uploaded resume.
- **Job Description Matching**: Calculates a percentage match between the resume and the job description.
- **Strengths & Weaknesses Analysis**: Uses AI to summarize key strengths and areas for improvement.
- **Web-based UI**: Users can upload resumes and enter job descriptions via a simple interface.
- **Email Feature**: Sends the results to the user's email using SMTP.

## Tech Stack
- **Backend**: Python, Flask
- **AI Processing**: Gemini API
- **Frontend**: HTML, CSS, JavaScript
- **Email Service**: SMTP (Gmail)

## Installation
### Prerequisites
- Python 3.x
- Flask
- Gemini API access
- SMTP Email credentials (Gmail)

### Steps to Install and Run
1. **Clone the repository:**
   ```sh
   https://github.com/sahilchalke0001/resume
   ```
2. **Create a virtual environment:**
   ```sh
   python -m venv venv
   source venv/bin/activate   # On Windows: venv\Scripts\activate
   ```
3. **Install dependencies:**
   ```sh
   pip install -r requirements.txt
   ```
4. **Set up environment variables:**
   Create a `.env` file and add the following:
   ```env
   GEMINI_API_KEY=your_api_key_here
   EMAIL_PASSWORD=your_email_password
   ```
5. **Run the Flask application:**
   ```sh
   python app.py
   ```
6. **Access the application:**
   Open `http://127.0.0.1:5000/` in your browser.

## Usage
1. Upload a resume (PDF).
2. Enter the job description.
3. Click **Analyze**.
4. View results on the website and receive them via email.

## Future Enhancements
- Support for multiple job descriptions.
- Advanced AI insights with additional NLP processing.
- Dashboard for tracking previous analyses.
- Integration with LinkedIn API for job suggestions.

## License
This project is licensed under the MIT License.

## Contact
For any queries, contact **sahilchalke@gmail.com.com**.

