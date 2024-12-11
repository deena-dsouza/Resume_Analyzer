from flask import Flask, render_template, request, send_file
import spacy
import PyPDF2
import docx
import os
import re
import csv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Load spaCy NER model for name entity recognition
nlp = spacy.load("en_core_web_sm")

# Ensure the uploads directory exists
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Configure the upload folder
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Supported file extensions
ALLOWED_EXTENSIONS = {'pdf', 'docx', 'txt'}

# Predefined list of technologies for various job roles
required_skills = {
    "Web Developer": ["HTML", "CSS", "JavaScript", "React", "Node.js", "PHP", "MySQL", "jQuery", "Bootstrap", "Vue.js"],
    "Data Scientist": ["Python", "R", "Machine Learning", "Deep Learning", "TensorFlow", "Keras", "Pandas", "NumPy", "Scikit-learn"],
    "Software Engineer": ["Java", "C++", "Python", "Algorithms", "Data Structures", "Git", "Spring", "Hibernate"],
    "Data Analyst":["Python", "R", "SQL","tableau","power BI"]
}

# Utility Functions
def allowed_file(filename):
    """Checks if the file is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_text_from_pdf(file_path):
    """Extracts text from a PDF file."""
    with open(file_path, "rb") as file:
        pdf_reader = PyPDF2.PdfReader(file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text

def extract_text_from_docx(file_path):
    """Extracts text from a DOCX file."""
    doc = docx.Document(file_path)
    # Join all paragraph texts into one string
    text = '\n'.join([paragraph.text for paragraph in doc.paragraphs])
    print("Extracted text from .docx:", text)  # Debugging: Print the extracted text
    return text

def extract_text_from_txt(file_path):
    """Extracts text from a TXT file."""
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()

def extract_text(file_path):
    """Extracts text based on file type."""
    if file_path.endswith('.pdf'):
        return extract_text_from_pdf(file_path)
    elif file_path.endswith('.docx'):
        return extract_text_from_docx(file_path)
    elif file_path.endswith('.txt'):
        return extract_text_from_txt(file_path)
    else:
        raise ValueError("Unsupported file format.")

def extract_entities(text):
    """Extracts names and emails from text."""
    emails = re.findall(r'\S+@\S+', text)
    # Improved regex to capture names (simple first and last names)
    names = re.findall(r'([A-Z][a-z]+(?:\s[A-Z][a-z]+)*)', text)  # Matches Firstname Lastname
    if names:
        names = [" ".join(names[0].split())]  # Join multiple first and last names if found
    else:
        names = []
    return emails, names

def get_skills_from_job_description(job_description):
    """Determines required skills based on the job description."""
    for role, skills in required_skills.items():
        if role.lower() in job_description.lower():
            return skills
    return required_skills.get("Web Developer", [])

# Routes
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/recruiter', methods=['GET', 'POST'])
def recruiter():
    results = []

    if request.method == 'POST':
        job_description = request.form.get('job_description', '')
        resume_files = request.files.getlist('resume_files')

        processed_resumes = []
        required = get_skills_from_job_description(job_description)

        for resume_file in resume_files:
            # Check if the file is allowed
            if allowed_file(resume_file.filename):
                sanitized_filename = secure_filename(resume_file.filename)
                file_path = os.path.join(UPLOAD_FOLDER, sanitized_filename)
                resume_file.save(file_path)

                # Extract text based on file type
                try:
                    resume_text = extract_text(file_path)
                except ValueError:
                    continue

                # Extract entities
                emails, names = extract_entities(resume_text)

                # Analyze present skills
                resume_words = set(resume_text.split())
                present_skills = [skill for skill in required if skill.lower() in (word.lower() for word in resume_words)]

                # TF-IDF ranking
                tfidf_vectorizer = TfidfVectorizer()
                job_desc_vector = tfidf_vectorizer.fit_transform([job_description])
                resume_vector = tfidf_vectorizer.transform([resume_text])
                similarity = cosine_similarity(job_desc_vector, resume_vector)[0][0] * 100  # Convert to percentage

                processed_resumes.append((names, emails, present_skills, similarity))

        # Sort by similarity
        processed_resumes.sort(key=lambda x: x[3], reverse=True)

        # Prepare results
        results = [
            (i + 1, resume[0], resume[1], resume[2], round(resume[3], 2))
            for i, resume in enumerate(processed_resumes)
        ]

        # Save results to a CSV file
        csv_file_path = os.path.join(app.config['UPLOAD_FOLDER'], 'ranked_resumes.csv')
        with open(csv_file_path, mode='w', newline='', encoding='utf-8') as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(['Rank', 'Name', 'Email', 'Skills Present', 'Similarity (%)'])
            for rank, names, emails, present_skills, similarity in results:
                csv_writer.writerow([rank, names[0] if names else "N/A", emails[0] if emails else "N/A",
                                     ", ".join(present_skills) if present_skills else "N/A", similarity])

    return render_template('recruiter.html', results=results)

@app.route('/seeker', methods=['GET', 'POST'])
def seeker():
    present_skills = []
    missing_skills = []

    if request.method == 'POST':
        job_description = request.form.get('job_description', '')
        resume_file = request.files.get('resume_file')

        if resume_file and allowed_file(resume_file.filename):
            sanitized_filename = secure_filename(resume_file.filename)
            file_path = os.path.join(UPLOAD_FOLDER, sanitized_filename)
            resume_file.save(file_path)

            # Extract text based on file type
            try:
                resume_text = extract_text(file_path)
            except ValueError:
                return "Unsupported file format.", 400

            # Analyze skills
            required = get_skills_from_job_description(job_description)
            resume_words = set(resume_text.split())
            present_skills = [skill for skill in required if skill.lower() in (word.lower() for word in resume_words)]
            missing_skills = [skill for skill in required if skill.lower() not in (word.lower() for word in resume_words)]

    return render_template('seeker.html', present_skills=present_skills, missing_skills=missing_skills)

@app.route('/download_csv')
def download_csv():
    """Download the ranking results as a CSV file."""
    csv_file_path = os.path.join(app.config['UPLOAD_FOLDER'], 'ranked_resumes.csv')
    if os.path.exists(csv_file_path):
        return send_file(csv_file_path, as_attachment=True)
    return "CSV file not found. Please analyze resumes first."

if __name__ == '__main__':
    app.run(debug=True)