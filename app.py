from flask import Flask, request, render_template, redirect, url_for, flash
import PyPDF2
import re
from collections import Counter
import nltk
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import spacy
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Enhanced list of fields and their related skills
FIELDS_AND_SKILLS = {
    'ML Engineer': [
        'python', 'r', 'java', 'c++', 'scala', 'julia',
        'machine learning', 'deep learning', 'neural networks', 'natural language processing',
        'computer vision', 'reinforcement learning', 'generative ai',
        'scikit-learn', 'tensorflow', 'keras', 'pytorch', 'mxnet', 'caffe',
        'pandas', 'numpy', 'scipy', 'matplotlib', 'seaborn', 'plotly',
        'data visualization', 'statistics', 'probability', 'linear algebra',
        'feature engineering', 'feature selection', 'dimensionality reduction',
        'model deployment', 'mlops', 'docker', 'kubernetes', 'jenkins',
        'aws', 'azure', 'gcp', 'cloud computing',
        'hadoop', 'spark', 'hive', 'pig', 'big data',
        'version control', 'git', 'github', 'gitlab',
        'agile', 'scrum', 'kanban', 'jira',
        'sql', 'nosql', 'mongodb', 'cassandra', 'redis',
        'restful apis', 'flask', 'django', 'fastapi',
        'ci/cd', 'unit testing', 'integration testing',
        'distributed systems', 'microservices architecture',
        'data structures', 'algorithms', 'system design',
        'model optimization', 'hyperparameter tuning', 'bayesian optimization',
        'time series analysis', 'anomaly detection', 'recommendation systems',
        'gpu programming', 'cuda', 'opencl',
        'ethics in ai', 'fairness in ml', 'explainable ai'
    ],
    'AI Engineer': [
        'python', 'java', 'c++', 'scala', 'lisp', 'prolog',
        'artificial intelligence', 'machine learning', 'deep learning',
        'natural language processing', 'nlp', 'computer vision', 'cv',
        'speech recognition', 'text-to-speech', 'chatbots',
        'pytorch', 'tensorflow', 'keras', 'caffe2', 'theano',
        'reinforcement learning', 'gans', 'transformers', 'bert', 'gpt',
        'object detection', 'image segmentation', 'facial recognition',
        'neural networks', 'cnn', 'rnn', 'lstm', 'gru',
        'ai ethics', 'explainable ai', 'interpretable ml',
        'knowledge representation', 'reasoning systems', 'expert systems',
        'robotics', 'autonomous systems', 'control systems',
        'graph neural networks', 'knowledge graphs',
        'bayesian networks', 'markov models', 'hidden markov models',
        'genetic algorithms', 'evolutionary computation',
        'fuzzy logic', 'swarm intelligence',
        'nlp libraries', 'spacy', 'nltk', 'gensim', 'hugging face',
        'opencv', 'pillow', 'scikit-image',
        'gpu acceleration', 'cuda', 'tensorrt',
        'distributed ai', 'federated learning',
        'ai-powered analytics', 'predictive modeling',
        'ai in iot', 'edge ai', 'ai on mobile devices',
        'quantum computing for ai', 'neuromorphic computing',
        'ai security', 'adversarial machine learning',
        'ai model compression', 'model quantization',
        'transfer learning', 'few-shot learning', 'zero-shot learning',
        'anomaly detection', 'fraud detection',
        'recommender systems', 'ranking algorithms'
    ],
    'Data Scientist': [
        'python', 'r', 'sas', 'matlab', 'scala', 'julia',
        'data analysis', 'statistical analysis', 'predictive modeling',
        'machine learning', 'deep learning', 'natural language processing',
        'data visualization', 'business intelligence',
        'pandas', 'numpy', 'scipy', 'scikit-learn', 'tensorflow', 'keras',
        'matplotlib', 'seaborn', 'plotly', 'tableau', 'power bi', 'qlik',
        'sql', 'nosql', 'mongodb', 'cassandra', 'neo4j',
        'big data', 'hadoop', 'spark', 'hive', 'impala',
        'data mining', 'text mining', 'web scraping',
        'a/b testing', 'hypothesis testing', 'experimental design',
        'regression analysis', 'classification', 'clustering',
        'time series analysis', 'forecasting', 'arima', 'sarima',
        'dimensionality reduction', 'pca', 't-sne', 'umap',
        'feature engineering', 'feature selection',
        'bayesian inference', 'monte carlo methods',
        'optimization algorithms', 'gradient descent',
        'data wrangling', 'data cleaning', 'etl processes',
        'database management', 'data modeling', 'data warehousing',
        'distributed computing', 'cloud platforms', 'aws', 'azure', 'gcp',
        'version control', 'git', 'github',
        'agile methodologies', 'scrum', 'kanban',
        'data ethics', 'data privacy', 'gdpr compliance',
        'storytelling with data', 'data-driven decision making',
        'geospatial analysis', 'gis',
        'survival analysis', 'cohort analysis',
        'recommendation systems', 'collaborative filtering',
        'anomaly detection', 'fraud detection',
        'causal inference', 'econometrics',
        'reinforcement learning', 'genetic algorithms',
        'neural networks', 'cnn', 'rnn', 'lstm',
        'nlp techniques', 'sentiment analysis', 'topic modeling',
        'computer vision', 'image processing'
    ],
    'Data Analyst': [
        'sql', 'mysql', 'postgresql', 'oracle', 'sql server',
        'python', 'r', 'sas', 'excel', 'vba',
        'data analysis', 'statistical analysis', 'descriptive statistics',
        'data visualization', 'tableau', 'power bi', 'qlik', 'looker',
        'excel', 'advanced excel', 'pivot tables', 'vlookup',
        'data cleaning', 'data wrangling', 'data quality assurance',
        'etl processes', 'data pipelines',
        'business intelligence', 'dashboarding', 'reporting',
        'kpi tracking', 'metrics definition',
        'pandas', 'numpy', 'matplotlib', 'seaborn',
        'statistical testing', 'hypothesis testing', 'a/b testing',
        'regression analysis', 'time series analysis',
        'forecasting', 'predictive analytics',
        'segmentation analysis', 'cluster analysis',
        'data modeling', 'dimensional modeling', 'star schema',
        'database design', 'data warehousing',
        'big data technologies', 'hadoop', 'hive', 'spark',
        'google analytics', 'web analytics', 'digital analytics',
        'social media analytics', 'marketing analytics',
        'financial analysis', 'budgeting', 'forecasting',
        'project management', 'agile methodologies',
        'data governance', 'data documentation',
        'data privacy', 'gdpr compliance',
        'data storytelling', 'presentation skills',
        'critical thinking', 'problem-solving',
        'business acumen', 'domain knowledge',
        'data mining', 'text analytics',
        'customer segmentation', 'churn analysis',
        'fraud detection', 'risk analysis',
        'supply chain analytics', 'inventory analysis',
        'process optimization', 'efficiency analysis',
        'geospatial analysis', 'mapping tools',
        'data scraping', 'api integration',
        'version control', 'git',
        'cloud platforms', 'aws', 'azure', 'gcp',
        'data ethics', 'responsible ai'
    ],
    'SQL Developer': [
        'sql', 'mysql', 'postgresql', 'oracle', 'sql server', 'db2',
        'pl/sql', 't-sql', 'transact-sql',
        'database management', 'database administration',
        'data modeling', 'er diagrams', 'normalization',
        'data warehousing', 'etl processes', 'data integration',
        'query optimization', 'performance tuning',
        'indexing strategies', 'query execution plans',
        'stored procedures', 'functions', 'triggers', 'views',
        'data migration', 'data conversion',
        'database security', 'user management', 'access control',
        'backup and recovery', 'disaster recovery',
        'high availability', 'replication',
        'nosql databases', 'mongodb', 'cassandra', 'redis',
        'big data technologies', 'hadoop', 'hive', 'impala',
        'data analysis', 'business intelligence',
        'reporting tools', 'crystal reports', 'ssrs',
        'etl tools', 'ssis', 'informatica', 'talend',
        'version control', 'git', 'svn',
        'agile methodologies', 'scrum',
        'python', 'java', 'c#', 'shell scripting',
        'xml', 'json', 'yaml',
        'api development', 'restful apis',
        'cloud databases', 'aws rds', 'azure sql', 'google cloud sql',
        'database design patterns',
        'data integrity', 'acid properties',
        'transaction management', 'concurrency control',
        'database sharding', 'partitioning',
        'olap', 'oltp', 'data cubes',
        'database testing', 'unit testing',
        'data quality management',
        'data governance', 'data compliance',
        'database documentation',
        'performance monitoring tools',
        'database auditing', 'log analysis',
        'data archiving', 'data retention policies',
        'database encryption', 'data masking'
    ],
    'Software Developer': [
        'java', 'python', 'c++', 'c#', 'javascript', 'typescript',
        'go', 'rust', 'swift', 'kotlin', 'scala', 'php', 'ruby',
        'html', 'css', 'sass', 'less',
        'react', 'angular', 'vue.js', 'svelte',
        'node.js', 'express.js', 'django', 'flask', 'spring', 'asp.net',
        'restful apis', 'graphql', 'microservices',
        'docker', 'kubernetes', 'jenkins', 'gitlab ci',
        'aws', 'azure', 'google cloud platform',
        'git', 'github', 'gitlab', 'bitbucket',
        'agile methodologies', 'scrum', 'kanban', 'xp',
        'test-driven development', 'behavior-driven development',
        'continuous integration', 'continuous deployment',
        'unit testing', 'integration testing', 'end-to-end testing',
        'jest', 'mocha', 'selenium', 'cypress',
        'design patterns', 'solid principles', 'clean code',
        'object-oriented programming', 'functional programming',
        'database management', 'sql', 'nosql',
        'orm frameworks', 'hibernate', 'entity framework',
        'caching mechanisms', 'redis', 'memcached',
        'message brokers', 'rabbitmq', 'apache kafka',
        'web security', 'oauth', 'jwt', 'https',
        'performance optimization', 'code profiling',
        'debugging tools', 'logging frameworks',
        'version control', 'code review practices',
        'ides', 'visual studio code', 'intellij idea', 'eclipse',
        'mobile app development', 'ios', 'android', 'react native',
        'progressive web apps', 'responsive design',
        'websockets', 'server-sent events',
        'blockchain development', 'smart contracts',
        'machine learning integration', 'ai apis',
        'serverless architecture', 'faas',
        'devops practices', 'infrastructure as code',
        'code quality tools', 'sonarqube', 'eslint',
        'package managers', 'npm', 'yarn', 'pip',
        'build tools', 'webpack', 'gulp', 'gradle',
        'virtualization', 'vmware', 'virtualbox',
        'networking fundamentals', 'tcp/ip', 'http/https',
        'data structures', 'algorithms', 'computational complexity',
        'system design', 'scalability', 'fault tolerance',
        'agile project management tools', 'jira', 'trello',
        'documentation', 'technical writing', 'api documentation'
    ]
}

def preprocess_text(text):
    text = re.sub(r'[^\w\s]', '', text.lower())
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

def extract_skills(text, skills_list):
    doc = nlp(text.lower())
    found_skills = []
    for token in doc:
        if token.text in skills_list or token.lemma_ in skills_list:
            found_skills.append(token.lemma_)
    return list(set(found_skills))

def calculate_similarity(text1, text2):
    text1 = preprocess_text(text1)
    text2 = preprocess_text(text2)
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([text1, text2])
    similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    return round(similarity * 100, 2)

def calculate_ats_score(resume_text, job_description_text, relevant_skills):
    preprocessed_resume = preprocess_text(resume_text)
    preprocessed_job = preprocess_text(job_description_text)
    
    resume_skills = extract_skills(preprocessed_resume, relevant_skills)
    job_skills = extract_skills(preprocessed_job, relevant_skills)
    
    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform([preprocessed_resume, preprocessed_job])
    
    resume_tfidf = tfidf_matrix[0].toarray()[0]
    job_tfidf = tfidf_matrix[1].toarray()[0]
    
    tfidf_score = sum(min(resume_tfidf[i], job_tfidf[i]) for i in range(len(tfidf.get_feature_names_out())))
    max_score = sum(job_tfidf)
    tfidf_percentage = (tfidf_score / max_score) * 100
    
    skill_match_percentage = len(set(resume_skills) & set(job_skills)) / len(set(job_skills)) * 100
    
    final_score = 0.7 * tfidf_percentage + 0.3 * skill_match_percentage
    
    similarity_score = calculate_similarity(resume_text, job_description_text)
    
    return round(final_score, 2), resume_skills, job_skills, similarity_score

def extract_text_from_pdf(pdf_file):
    reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in range(len(reader.pages)):
        text += reader.pages[page].extract_text()
    return text

@app.route('/', methods=['GET', 'POST'])
def index():
    fields = list(FIELDS_AND_SKILLS.keys())
    
    if request.method == 'POST':
        if 'resume' not in request.files or 'job_description' not in request.form or 'field' not in request.form:
            flash('Please fill out all fields', 'error')
            return redirect(request.url)
        
        pdf_file = request.files['resume']
        job_description_text = request.form['job_description']
        selected_field = request.form['field']
        
        if pdf_file.filename == '':
            flash('No file selected', 'error')
            return redirect(request.url)
        
        if pdf_file and pdf_file.filename.endswith('.pdf'):
            try:
                resume_text = extract_text_from_pdf(pdf_file)
                relevant_skills = FIELDS_AND_SKILLS.get(selected_field, [])
                ats_score, resume_skills, job_skills, similarity_score = calculate_ats_score(resume_text, job_description_text, relevant_skills)
                
                matched_skills = list(set(resume_skills) & set(job_skills))
                missing_skills = list(set(job_skills) - set(resume_skills))
                
                return render_template('result.html', 
                                       ats_score=ats_score, 
                                       missing_skills=missing_skills, 
                                       matched_skills=matched_skills,
                                       resume_skills=resume_skills,
                                       job_skills=job_skills,
                                       selected_field=selected_field,
                                       similarity_score=similarity_score)
            except Exception as e:
                flash(f'An error occurred: {str(e)}', 'error')
                return redirect(request.url)
        else:
            flash('Please upload a PDF file', 'error')
            return redirect(request.url)
    
    return render_template('index.html', fields=fields)

@app.route('/result')
def result():
    return render_template('result.html')

if __name__ == '__main__':
    app.run(debug=True)