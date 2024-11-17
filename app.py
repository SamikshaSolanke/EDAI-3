from flask import Flask, request, render_template
from PyPDF2 import PdfReader
import pickle
import os
import re
import spacy
import google.generativeai as genai
import textwrap
from IPython.display import Markdown
from kiwisolver import strength

nlp = spacy.load('en_core_web_sm')

API_KEY = "AIzaSyDjHvCOZg3RaZt8cJoo4D0KRARceRvp_kU"

app = Flask(__name__)

# Load models and vectorizers
rf_classifier_categorization = pickle.load(open('models/rf_classifier_categorization.pkl', 'rb'))
tfidf_vectorizer_categorization = pickle.load(open('models/tfidf_vectorizer_categorization.pkl', 'rb'))
rf_classifier_job_recommendation = pickle.load(open('models/rf_classifier_job_recommendation.pkl', 'rb'))
tfidf_vectorizer_job_recommendation = pickle.load(open('models/tfidf_vectorizer_job_recommendation.pkl', 'rb'))


# Resume text cleaning function
def cleanresume(txt):
    cleanText = re.sub(r'http\S+\s', ' ', txt)
    cleanText = re.sub(r'RT|cc', ' ', cleanText)
    cleanText = re.sub(r'#\S+\s', ' ', cleanText)
    cleanText = re.sub(r'@\S+', '  ', cleanText)
    cleanText = re.sub(r'[%s]' % re.escape(r"""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', cleanText)
    cleanText = re.sub(r'[^\x00-\x7f]', ' ', cleanText)
    cleanText = re.sub(r'\s+', ' ', cleanText)
    return cleanText


# def get_strengths_and_weaknesses(text):
#     # Configure the generative AI API
#     genai.configure(api_key=API_KEY)
#
#     # Define the generative model to use
#     modelai = genai.GenerativeModel('gemini-pro')
#
#     # Generate the content for strengths
#     strength_res = modelai.generate_content(
#         prompt="Analyze the following text and summarize the candidate's strengths in a concise manner:",
#         input_text=text
#     )
#
#     # Generate the content for weaknesses
#     weakness_res = modelai.generate_content(
#         prompt="Analyze the following text and summarize the candidate's weaknesses in a concise manner:",
#         input_text=text
#     )
#
#     # Function to convert response to Markdown
#     def to_markdown(text):
#         lines = text.split('. ')
#         formatted_text = "\n\n".join([f"* {line.strip()}" for line in lines if line.strip()])
#         return Markdown(formatted_text)
#
#     # Convert the responses to Markdown format
#     strengths = to_markdown(strength_res)
#     weaknesses = to_markdown(weakness_res)
#
#     return strengths, weaknesses

def get_strengths_and_weaknesses(text):
    # Configure the generative AI API
    genai.configure(api_key=API_KEY)

    # Define the generative model to use
    modelai = genai.GenerativeModel('gemini-pro')

    # Generate the content for strengths
    strength_res = modelai.generate_content("Analyze the following text and summarize the candidate's strengths in a concise manner:",)

    # Generate the content for weaknesses
    weakness_res = modelai.generate_content("Analyze the following text and summarize the candidate's weaknesses in a concise manner:",)

    # Function to convert response to Markdown
    def to_markdown(text):
        text = text.replace('.', ' *')
        return Markdown(textwrap.indent(text, '>', predicate=lambda _: True))

    # Convert the responses to Markdown format
    strengths = to_markdown(strength_res.text)
    weaknesses = to_markdown(weakness_res.text)

    return strengths, weaknesses

# Predict resume category
def predict_category(resume_text):
    resume_text = cleanresume(resume_text)
    resume_tfidf = tfidf_vectorizer_categorization.transform([resume_text])
    predicted_category = rf_classifier_categorization.predict(resume_tfidf)[0]
    return predicted_category


# Recommend job based on resume
def job_recommendation(resume_text):
    resume_text = cleanresume(resume_text)
    resume_tfidf = tfidf_vectorizer_job_recommendation.transform([resume_text])
    recommended_job = rf_classifier_job_recommendation.predict(resume_tfidf)[0]
    return recommended_job


# Convert PDF to text
def pdf_to_text(file):
    reader = PdfReader(file)
    text = ''
    for page in range(len(reader.pages)):
        text += reader.pages[page].extract_text()
    return text


# Extract contact number from resume
def extract_contact_number_from_resume(text):
    pattern = r"\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"
    match = re.search(pattern, text)
    return match.group() if match else None


# Extract email from resume
def extract_email_from_resume(text):
    pattern = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
    match = re.search(pattern, text)
    return match.group() if match else None


# Extract skills from resume
def extract_skills_from_resume(text):
    skills_list = [
        'Python', 'Data Analysis', 'Machine Learning', 'Communication', 'Project Management', 'Deep Learning', 'SQL',
        'Tableau',
        'Java', 'C++', 'JavaScript', 'HTML', 'CSS', 'React', 'Angular', 'Node.js', 'MongoDB', 'Express.js', 'Git',
        'Research', 'Statistics', 'Quantitative Analysis', 'Qualitative Analysis', 'SPSS', 'R', 'Data Visualization',
        'Matplotlib',
        'Seaborn', 'Plotly', 'Pandas', 'Numpy', 'Scikit-learn', 'TensorFlow', 'Keras', 'PyTorch', 'NLTK', 'Text Mining',
        'Natural Language Processing', 'Computer Vision', 'Image Processing', 'OCR', 'Speech Recognition',
        'Recommendation Systems',
        'Collaborative Filtering', 'Content-Based Filtering', 'Reinforcement Learning', 'Neural Networks',
        'Convolutional Neural Networks',
        'Recurrent Neural Networks', 'Generative Adversarial Networks', 'XGBoost', 'Random Forest', 'Decision Trees',
        'Support Vector Machines',
        'Linear Regression', 'Logistic Regression', 'K-Means Clustering', 'Hierarchical Clustering', 'DBSCAN',
        'Association Rule Learning',
        'Apache Hadoop', 'Apache Spark', 'MapReduce', 'Hive', 'HBase', 'Apache Kafka', 'Data Warehousing', 'ETL',
        'Big Data Analytics',
        'Cloud Computing', 'Amazon Web Services (AWS)', 'Microsoft Azure', 'Google Cloud Platform (GCP)', 'Docker',
        'Kubernetes', 'Linux',
        'Shell Scripting', 'Cybersecurity', 'Network Security', 'Penetration Testing', 'Firewalls', 'Encryption',
        'Malware Analysis',
        'Digital Forensics', 'CI/CD', 'DevOps', 'Agile Methodology', 'Scrum', 'Kanban', 'Continuous Integration',
        'Continuous Deployment',
        'Software Development', 'Web Development', 'Mobile Development', 'Backend Development', 'Frontend Development',
        'Full-Stack Development',
        'UI/UX Design', 'Responsive Design', 'Wireframing', 'Prototyping', 'User Testing', 'Adobe Creative Suite',
        'Photoshop', 'Illustrator',
        'InDesign', 'Figma', 'Sketch', 'Zeplin', 'InVision', 'Product Management', 'Market Research',
        'Customer Development', 'Lean Startup',
        'Business Development', 'Sales', 'Marketing', 'Content Marketing', 'Social Media Marketing', 'Email Marketing',
        'SEO', 'SEM', 'PPC',
        'Google Analytics', 'Facebook Ads', 'LinkedIn Ads', 'Lead Generation', 'Customer Relationship Management (CRM)',
        'Salesforce',
        'HubSpot', 'Zendesk', 'Intercom', 'Customer Support', 'Technical Support', 'Troubleshooting',
        'Ticketing Systems', 'ServiceNow',
        'ITIL', 'Quality Assurance', 'Manual Testing', 'Automated Testing', 'Selenium', 'JUnit', 'Load Testing',
        'Performance Testing',
        'Regression Testing', 'Black Box Testing', 'White Box Testing', 'API Testing', 'Mobile Testing',
        'Usability Testing', 'Accessibility Testing',
        'Cross-Browser Testing', 'Agile Testing', 'User Acceptance Testing', 'Software Documentation',
        'Technical Writing', 'Copywriting',
        'Editing', 'Proofreading', 'Content Management Systems (CMS)', 'WordPress', 'Joomla', 'Drupal', 'Magento',
        'Shopify', 'E-commerce',
        'Payment Gateways', 'Inventory Management', 'Supply Chain Management', 'Logistics', 'Procurement',
        'ERP Systems', 'SAP', 'Oracle',
        'Microsoft Dynamics', 'Tableau', 'Power BI', 'QlikView', 'Looker', 'Data Warehousing', 'ETL',
        'Data Engineering', 'Data Governance',
        'Data Quality', 'Master Data Management', 'Predictive Analytics', 'Prescriptive Analytics',
        'Descriptive Analytics', 'Business Intelligence',
        'Dashboarding', 'Reporting', 'Data Mining', 'Web Scraping', 'API Integration', 'RESTful APIs', 'GraphQL',
        'SOAP', 'Microservices',
        'Serverless Architecture', 'Lambda Functions', 'Event-Driven Architecture', 'Message Queues', 'GraphQL',
        'Socket.io', 'WebSockets'
                     'Ruby', 'Ruby on Rails', 'PHP', 'Symfony', 'Laravel', 'CakePHP', 'Zend Framework', 'ASP.NET', 'C#',
        'VB.NET', 'ASP.NET MVC', 'Entity Framework',
        'Spring', 'Hibernate', 'Struts', 'Kotlin', 'Swift', 'Objective-C', 'iOS Development', 'Android Development',
        'Flutter', 'React Native', 'Ionic',
        'Mobile UI/UX Design', 'Material Design', 'SwiftUI', 'RxJava', 'RxSwift', 'Django', 'Flask', 'FastAPI',
        'Falcon', 'Tornado', 'WebSockets',
        'GraphQL', 'RESTful Web Services', 'SOAP', 'Microservices Architecture', 'Serverless Computing', 'AWS Lambda',
        'Google Cloud Functions',
        'Azure Functions', 'Server Administration', 'System Administration', 'Network Administration',
        'Database Administration', 'MySQL', 'PostgreSQL',
        'SQLite', 'Microsoft SQL Server', 'Oracle Database', 'NoSQL', 'MongoDB', 'Cassandra', 'Redis', 'Elasticsearch',
        'Firebase', 'Google Analytics',
        'Google Tag Manager', 'Adobe Analytics', 'Marketing Automation', 'Customer Data Platforms', 'Segment',
        'Salesforce Marketing Cloud', 'HubSpot CRM',
        'Zapier', 'IFTTT', 'Workflow Automation', 'Robotic Process Automation (RPA)', 'UI Automation',
        'Natural Language Generation (NLG)',
        'Virtual Reality (VR)', 'Augmented Reality (AR)', 'Mixed Reality (MR)', 'Unity', 'Unreal Engine', '3D Modeling',
        'Animation', 'Motion Graphics',
        'Game Design', 'Game Development', 'Level Design', 'Unity3D', 'Unreal Engine 4', 'Blender', 'Maya',
        'Adobe After Effects', 'Adobe Premiere Pro',
        'Final Cut Pro', 'Video Editing', 'Audio Editing', 'Sound Design', 'Music Production', 'Digital Marketing',
        'Content Strategy', 'Conversion Rate Optimization (CRO)',
        'A/B Testing', 'Customer Experience (CX)', 'User Experience (UX)', 'User Interface (UI)', 'Persona Development',
        'User Journey Mapping', 'Information Architecture (IA)',
        'Wireframing', 'Prototyping', 'Usability Testing', 'Accessibility Compliance', 'Internationalization (I18n)',
        'Localization (L10n)', 'Voice User Interface (VUI)',
        'Chatbots', 'Natural Language Understanding (NLU)', 'Speech Synthesis', 'Emotion Detection',
        'Sentiment Analysis', 'Image Recognition', 'Object Detection',
        'Facial Recognition', 'Gesture Recognition', 'Document Recognition', 'Fraud Detection',
        'Cyber Threat Intelligence', 'Security Information and Event Management (SIEM)',
        'Vulnerability Assessment', 'Incident Response', 'Forensic Analysis', 'Security Operations Center (SOC)',
        'Identity and Access Management (IAM)', 'Single Sign-On (SSO)',
        'Multi-Factor Authentication (MFA)', 'Blockchain', 'Cryptocurrency', 'Decentralized Finance (DeFi)',
        'Smart Contracts', 'Web3', 'Non-Fungible Tokens (NFTs)']

    skills = [skill for skill in skills_list if re.search(r"\b{}\b".format(re.escape(skill)), text, re.IGNORECASE)]
    return skills


# Extract education from resume
def extract_education_from_resume(text):
    education_keywords = [
        'Computer Science', 'Information Technology', 'Software Engineering', 'Electrical Engineering',
        'Mechanical Engineering', 'Civil Engineering',
        'Chemical Engineering', 'Biomedical Engineering', 'Aerospace Engineering', 'Nuclear Engineering',
        'Industrial Engineering', 'Systems Engineering',
        'Environmental Engineering', 'Petroleum Engineering', 'Geological Engineering', 'Marine Engineering',
        'Robotics Engineering', 'Biotechnology',
        'Biochemistry', 'Microbiology', 'Genetics', 'Molecular Biology', 'Bioinformatics', 'Neuroscience', 'Biophysics',
        'Biostatistics', 'Pharmacology',
        'Physiology', 'Anatomy', 'Pathology', 'Immunology', 'Epidemiology', 'Public Health', 'Health Administration',
        'Nursing', 'Medicine', 'Dentistry',
        'Pharmacy', 'Veterinary Medicine', 'Medical Technology', 'Radiography', 'Physical Therapy',
        'Occupational Therapy', 'Speech Therapy', 'Nutrition',
        'Sports Science', 'Kinesiology', 'Exercise Physiology', 'Sports Medicine', 'Rehabilitation Science',
        'Psychology', 'Counseling', 'Social Work',
        'Sociology', 'Anthropology', 'Criminal Justice', 'Political Science', 'International Relations', 'Economics',
        'Finance', 'Accounting', 'Business Administration',
        'Management', 'Marketing', 'Entrepreneurship', 'Hospitality Management', 'Tourism Management',
        'Supply Chain Management', 'Logistics Management',
        'Operations Management', 'Human Resource Management', 'Organizational Behavior', 'Project Management',
        'Quality Management', 'Risk Management',
        'Strategic Management', 'Public Administration', 'Urban Planning', 'Architecture', 'Interior Design',
        'Landscape Architecture', 'Fine Arts',
        'Visual Arts', 'Graphic Design', 'Fashion Design', 'Industrial Design', 'Product Design', 'Animation',
        'Film Studies', 'Media Studies',
        'Communication Studies', 'Journalism', 'Broadcasting', 'Creative Writing', 'English Literature', 'Linguistics',
        'Translation Studies',
        'Foreign Languages', 'Modern Languages', 'Classical Studies', 'History', 'Archaeology', 'Philosophy',
        'Theology', 'Religious Studies',
        'Ethics', 'Education', 'Early Childhood Education', 'Elementary Education', 'Secondary Education',
        'Special Education', 'Higher Education',
        'Adult Education', 'Distance Education', 'Online Education', 'Instructional Design', 'Curriculum Development',
        'Library Science',
        'Information Science', 'Computer Engineering', 'Software Development', 'Cybersecurity', 'Information Security',
        'Network Engineering', 'Data Science', 'Data Analytics', 'Business Analytics', 'Operations Research',
        'Decision Sciences',
        'Human-Computer Interaction', 'User Experience Design', 'User Interface Design', 'Digital Marketing',
        'Content Strategy',
        'Brand Management', 'Public Relations', 'Corporate Communications', 'Media Production', 'Digital Media',
        'Web Development',
        'Mobile App Development', 'Game Development', 'Virtual Reality', 'Augmented Reality', 'Blockchain Technology',
        'Cryptocurrency',
        'Digital Forensics', 'Forensic Science', 'Criminalistics', 'Crime Scene Investigation', 'Emergency Management',
        'Fire Science',
        'Environmental Science', 'Climate Science', 'Meteorology', 'Geography', 'Geomatics', 'Remote Sensing',
        'Geoinformatics',
        'Cartography', 'GIS (Geographic Information Systems)', 'Environmental Management', 'Sustainability Studies',
        'Renewable Energy',
        'Green Technology', 'Ecology', 'Conservation Biology', 'Wildlife Biology', 'Zoology']

    education = [keyword for keyword in education_keywords if re.search(r"(?i)\b{}\b".format(re.escape(keyword)), text)]
    return education


# Extract name from resume option 1
def extract_name_from_resume(text):
    pattern = r'\b[A-Z][a-z]*\s[A-Z][a-z]+\b|\b[A-Z][a-z]+\s[A-Z][a-z]*\b'
    matches = re.findall(pattern, text)
    return matches[0] if matches else None

# extraction of name based on position of title
# def extract_name_from_resume(text):
#     # Consider only the first few lines of the text where names are typically located
#     lines = text.split('\n')
#     top_lines = " ".join(lines[:10])  # Check first 10 lines
#
#     # Apply regex to extract name
#     pattern = r'\b[A-Z][a-z]+(?:\s[A-Z][a-z]+)+\b'
#     matches = re.findall(pattern, top_lines)
#
#     return matches[0] if matches else None


# Score the resume based on skills, education, and experience
def score_resume(skills, education, experience):
    skill_score = len(skills) * 2  # Each skill adds 2 points
    education_score = len(education) * 3  # Each education entry adds 3 points
    experience_score = experience * 5  # Each experience entry adds 5 points
    total_score = skill_score + education_score + experience_score

    if total_score >= 50:
        return "strong"
    elif 20 <= total_score < 50:
        return "medium"
    else:
        return "weak"


# Routes
@app.route('/')
def resume():
    return render_template("resume.html")


@app.route('/pred', methods=['POST'])
def pred():
    if 'resume' in request.files:
        file = request.files['resume']
        filename = file.filename
        if filename.endswith('.pdf'):
            text = pdf_to_text(file)
        elif filename.endswith('.txt'):
            text = file.read().decode('utf-8')
        else:
            return render_template('resume.html', message="Invalid file format. Please upload a PDF or TXT file.")

        predicted_category = predict_category(text)
        recommended_job = job_recommendation(text)
        phone = extract_contact_number_from_resume(text)
        email = extract_email_from_resume(text)
        extracted_skills = extract_skills_from_resume(text)
        extracted_education = extract_education_from_resume(text)
        name = extract_name_from_resume(text)
        strengths, weaknesses = get_strengths_and_weaknesses(text)
        # Assuming experience is based on the length of education entries as a placeholder
        experience = len(extracted_education)
        resume_score = score_resume(extracted_skills, extracted_education, experience)

        return render_template('resume.html', predicted_category=predicted_category, recommended_job=recommended_job,
                               phone=phone, name=name, email=email, extracted_skills=extracted_skills,
                               extracted_education=extracted_education, resume_score=resume_score, strengths=strengths, weaknesses=weaknesses)
    else:
        return render_template("resume.html", message="No resume file uploaded.")


if __name__ == '__main__':
    app.run(debug=True)

# from flask import Flask, request, render_template
# from PyPDF2 import PdfReader
# import pickle
# import os
# import re
# from transformers import pipeline
#
# app = Flask(__name__)
#
# # Load pre-trained models
# rf_classifier_categorization = pickle.load(open('models/rf_classifier_categorization.pkl', 'rb'))
# tfidf_vectorizer_categorization = pickle.load(open('models/tfidf_vectorizer_categorization.pkl', 'rb'))
# rf_classifier_job_recommendation = pickle.load(open('models/rf_classifier_job_recommendation.pkl', 'rb'))
# tfidf_vectorizer_job_recommendation = pickle.load(open('models/tfidf_vectorizer_job_recommendation.pkl', 'rb'))
#
# # Initialize the Hugging Face summarization model (or text classifier for specific strengths/weaknesses)
# summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6", timeout=300)
#
#
# def cleanresume(txt):
#     cleanText = re.sub(r'http\S+\s', ' ', txt)
#     cleanText = re.sub(r'RT|cc', ' ', cleanText)
#     cleanText = re.sub(r'#\S+\s', ' ', cleanText)
#     cleanText = re.sub(r'@\S+', '  ', cleanText)
#     cleanText = re.sub(r'[%s]' % re.escape(r"""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', cleanText)
#     cleanText = re.sub(r'[^\x00-\x7f]', ' ', cleanText)
#     cleanText = re.sub(r'\s+', ' ', cleanText)
#     return cleanText
#
#
# def predict_category(resume_text):
#     resume_text = cleanresume(resume_text)
#     resume_tfidf = tfidf_vectorizer_categorization.transform([resume_text])
#     predicted_category = rf_classifier_categorization.predict(resume_tfidf)[0]
#     return predicted_category
#
#
# def job_recommendation(resume_text):
#     resume_text = cleanresume(resume_text)
#     resume_tfidf = tfidf_vectorizer_job_recommendation.transform([resume_text])
#     recommended_job = rf_classifier_job_recommendation.predict(resume_tfidf)[0]
#     return recommended_job
#
#
# def pdf_to_text(file):
#     reader = PdfReader(file)
#     text = ''
#     for page in range(len(reader.pages)):
#         text += reader.pages[page].extract_text()
#     return text
#
#
# # Strengths and weaknesses extraction using Hugging Face summarizer
# def extract_strengths_and_weaknesses(text):
#     summary = summarizer(text, max_length=130, min_length=30, do_sample=False)
#     summarized_text = summary[0]['summary_text']
#
#     # You can refine this part by classifying specific keywords as strengths or weaknesses
#     # For simplicity, we'll assume the summary contains both
#     strengths = summarized_text[:len(summarized_text) // 2]  # First half as strengths
#     weaknesses = summarized_text[len(summarized_text) // 2:]  # Second half as weaknesses
#     return strengths, weaknesses
#
#
# def extract_contact_number_from_resume(text):
#     contact_number = None
#     pattern = r"\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"
#     match = re.search(pattern, text)
#     if match:
#         contact_number = match.group()
#     return contact_number
#
#
# def extract_email_from_resume(text):
#     email = None
#     pattern = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
#     match = re.search(pattern, text)
#     if match:
#         email = match.group()
#     return email
#
#
# def score_resume(skills, education, experience):
#     skill_score = len(skills) * 2  # Example: 2 points per skill
#     education_score = len(education) * 3  # Example: 3 points per education qualification
#     experience_score = experience  # Assuming experience is already quantified (e.g., years)
#     total_score = skill_score + education_score + experience_score
#     if total_score > 50:
#         return "strong"
#     elif total_score > 20:
#         return "medium"
#     else:
#         return "weak"
#
#
# def extract_skills_from_resume(text):
#     skills_list = ['Python', 'Data Analysis', 'Machine Learning', 'Communication', 'Project Management', 'SQL', 'Java',
#                    'C++']
#     skills = []
#     for skill in skills_list:
#         pattern = r"\b{}\b".format(re.escape(skill))
#         match = re.search(pattern, text, re.IGNORECASE)
#         if match:
#             skills.append(skill)
#     return skills
#
#
# def extract_education_from_resume(text):
#     education = []
#     education_keywords = ['Computer Science', 'Information Technology', 'Software Engineering']
#     for keyword in education_keywords:
#         pattern = r"(?i)\b{}\b".format(re.escape(keyword))
#         match = re.search(pattern, text)
#         if match:
#             education.append(match.group())
#     return education
#
#
# def extract_name_from_resume(text):
#     name = None
#     pattern = r'\b[A-Z][a-z]*\s[A-Z][a-z]+\b|\b[A-Z][a-z]+\s[A-Z][a-z]*\b'
#     matches = re.findall(pattern, text)
#     if matches:
#         name = matches[0]
#     return name
#
#
# # routes===============================================
#
# @app.route('/')
# def resume():
#     return render_template("resume.html")
#
#
# @app.route('/pred', methods=['POST'])
# def pred():
#     if 'resume' in request.files:
#         file = request.files['resume']
#         filename = file.filename
#         if filename.endswith('.pdf'):
#             text = pdf_to_text(file)
#         elif filename.endswith('.txt'):
#             text = file.read().decode('utf-8')
#         else:
#             return render_template('resume.html', message="Invalid file format. Please upload a PDF or TXT file.")
#
#         predicted_category = predict_category(text)
#         recommended_job = job_recommendation(text)
#         phone = extract_contact_number_from_resume(text)
#         email = extract_email_from_resume(text)
#         extracted_skills = extract_skills_from_resume(text)
#         extracted_education = extract_education_from_resume(text)
#         name = extract_name_from_resume(text)
#         experience = len(extracted_education)
#         resume_score = score_resume(extracted_skills, extracted_education, experience)
#
#         # Extract strengths and weaknesses
#         strengths, weaknesses = extract_strengths_and_weaknesses(text)
#
#         return render_template('resume.html', predicted_category=predicted_category, recommended_job=recommended_job,
#                                phone=phone, name=name, email=email, extracted_skills=extracted_skills,
#                                extracted_education=extracted_education, resume_score=resume_score,
#                                strengths=strengths, weaknesses=weaknesses)
#     else:
#         return render_template("resume.html", message="No resume file uploaded.")
#
#
# if __name__ == '__main__':
#     app.run(debug=True)
#
