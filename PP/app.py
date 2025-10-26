import os
import sqlite3
import hashlib
import secrets
from datetime import datetime, timedelta
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import pickle
import json

# Flask and related imports
try:
    from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
    from werkzeug.security import generate_password_hash, check_password_hash
    FLASK_AVAILABLE = True
except ImportError:
    print("Flask not available. This is a demonstration of the application structure.")
    FLASK_AVAILABLE = False

class PersonalityPredictor:
    """Enhanced personality prediction system"""
    
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=500, stop_words='english')
        self.models = {}
        self.scaler = StandardScaler()
        self.personality_traits = ['openness', 'conscientiousness', 'extraversion', 'agreeableness', 'neuroticism']
        self.is_trained = False
        
        # Personality-related keywords
        self.trait_keywords = {
            'openness': ['creative', 'imaginative', 'artistic', 'curious', 'original', 'innovative', 'abstract', 'complex', 'intellectual', 'philosophical'],
            'conscientiousness': ['organized', 'responsible', 'reliable', 'disciplined', 'careful', 'thorough', 'systematic', 'efficient', 'planned', 'structured'],
            'extraversion': ['outgoing', 'social', 'energetic', 'talkative', 'assertive', 'active', 'enthusiastic', 'friendly', 'party', 'people'],
            'agreeableness': ['kind', 'sympathetic', 'cooperative', 'trusting', 'helpful', 'forgiving', 'generous', 'considerate', 'empathetic', 'caring'],
            'neuroticism': ['anxious', 'worried', 'nervous', 'stressed', 'emotional', 'unstable', 'moody', 'tense', 'fearful', 'insecure']
        }
        
        self._train_initial_model()
    
    def extract_text_features(self, text):
        """Extract features from text"""
        features = {}
        words = text.lower().split()
        
        # Basic statistics
        features['word_count'] = len(words)
        features['avg_word_length'] = np.mean([len(word) for word in words]) if words else 0
        features['sentence_count'] = len([s for s in text.split('.') if s.strip()])
        
        # Emotional indicators
        positive_words = ['good', 'great', 'excellent', 'amazing', 'wonderful', 'love', 'like', 'enjoy', 'happy', 'excited']
        negative_words = ['bad', 'terrible', 'awful', 'hate', 'dislike', 'sad', 'angry', 'frustrated', 'disappointed', 'worried']
        
        features['positive_ratio'] = sum(1 for word in words if word in positive_words) / len(words) if words else 0
        features['negative_ratio'] = sum(1 for word in words if word in negative_words) / len(words) if words else 0
        
        # Trait-specific word ratios
        for trait, keywords in self.trait_keywords.items():
            features[f'{trait}_ratio'] = sum(1 for word in words if word in keywords) / len(words) if words else 0
        
        # Punctuation usage
        features['exclamation_ratio'] = text.count('!') / len(text) if text else 0
        features['question_ratio'] = text.count('?') / len(text) if text else 0
        
        return list(features.values())
    
    def _train_initial_model(self):
        """Train initial model with synthetic data"""
        # Create synthetic training data
        training_texts = [
            "I love exploring new ideas and being creative. Art and philosophy fascinate me.",
            "I always organize my schedule carefully and complete tasks on time. Planning is essential.",
            "I enjoy meeting people and going to social events. Parties energize me!",
            "I try to help others and believe in cooperation. Kindness matters most.",
            "I often worry about things and feel anxious about the future.",
            "Reading books and learning new concepts excites me. I love abstract thinking.",
            "I keep detailed lists and follow strict routines. Everything has its place.",
            "I'm energetic and love being around friends. Let's have adventures!",
            "Understanding others is important. I trust people and forgive easily.",
            "I get stressed easily and sometimes feel overwhelmed by emotions.",
        ] * 20  # Repeat for more training data
        
        # Generate corresponding personality scores
        training_scores = []
        for text in training_texts:
            scores = {}
            # Simple rule-based scoring for demonstration
            if any(word in text.lower() for word in ['creative', 'art', 'philosophy', 'abstract']):
                scores['openness'] = np.random.normal(0.8, 0.1)
            else:
                scores['openness'] = np.random.normal(0.4, 0.2)
            
            if any(word in text.lower() for word in ['organize', 'schedule', 'planning', 'routine']):
                scores['conscientiousness'] = np.random.normal(0.8, 0.1)
            else:
                scores['conscientiousness'] = np.random.normal(0.5, 0.2)
            
            if any(word in text.lower() for word in ['social', 'party', 'energetic', 'friends']):
                scores['extraversion'] = np.random.normal(0.8, 0.1)
            else:
                scores['extraversion'] = np.random.normal(0.4, 0.2)
            
            if any(word in text.lower() for word in ['help', 'cooperation', 'kindness', 'trust']):
                scores['agreeableness'] = np.random.normal(0.8, 0.1)
            else:
                scores['agreeableness'] = np.random.normal(0.5, 0.2)
            
            if any(word in text.lower() for word in ['worry', 'anxious', 'stress', 'overwhelmed']):
                scores['neuroticism'] = np.random.normal(0.7, 0.1)
            else:
                scores['neuroticism'] = np.random.normal(0.3, 0.2)
            
            # Clip values to [0, 1]
            for trait in scores:
                scores[trait] = np.clip(scores[trait], 0, 1)
            
            training_scores.append(scores)
        
        # Extract features
        text_features = [self.extract_text_features(text) for text in training_texts]
        tfidf_features = self.vectorizer.fit_transform(training_texts).toarray()
        
        # Combine features
        combined_features = np.hstack([tfidf_features, text_features])
        combined_features = self.scaler.fit_transform(combined_features)
        
        # Train models for each trait
        for trait in self.personality_traits:
            y = [scores[trait] for scores in training_scores]
            model = RandomForestRegressor(n_estimators=50, random_state=42)
            model.fit(combined_features, y)
            self.models[trait] = model
        
        self.is_trained = True
        print("Personality prediction model trained successfully!")
    
    def predict_personality(self, text):
        """Predict personality from text"""
        if not self.is_trained:
            return None
        
        # Extract features
        text_features = self.extract_text_features(text)
        tfidf_features = self.vectorizer.transform([text]).toarray()
        
        # Combine features
        combined_features = np.hstack([tfidf_features, [text_features]])
        combined_features = self.scaler.transform(combined_features)
        
        # Make predictions
        predictions = {}
        for trait in self.personality_traits:
            score = self.models[trait].predict(combined_features)[0]
            predictions[trait] = np.clip(score, 0, 1)
        
        return predictions

class DatabaseManager:
    """Database management for the application"""
    
    def __init__(self, db_path='personality_app.db'):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize database tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Users table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                is_admin BOOLEAN DEFAULT FALSE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_login TIMESTAMP
            )
        ''')
        
        # Questions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS questions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                question_text TEXT NOT NULL,
                question_type TEXT DEFAULT 'text',
                is_active BOOLEAN DEFAULT TRUE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # User responses table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_responses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                question_id INTEGER,
                response_text TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id),
                FOREIGN KEY (question_id) REFERENCES questions (id)
            )
        ''')
        
        # Personality results table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS personality_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                openness REAL,
                conscientiousness REAL,
                extraversion REAL,
                agreeableness REAL,
                neuroticism REAL,
                combined_text TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')
        
        conn.commit()
        
        # Insert default questions if none exist
        cursor.execute('SELECT COUNT(*) FROM questions')
        if cursor.fetchone()[0] == 0:
            default_questions = [
                "Describe yourself in a few sentences. What are your main interests and hobbies?",
                "How do you typically spend your free time? What activities do you enjoy most?",
                "Describe a challenging situation you faced recently and how you handled it.",
                "What motivates you in life? What are your goals and aspirations?",
                "How do you interact with others? Describe your social preferences.",
                "What kind of work environment do you prefer? How do you approach tasks?",
                "Describe your emotional responses. How do you handle stress and pressure?",
                "What are your thoughts on trying new experiences and taking risks?",
                "How do you make decisions? Do you prefer planning or spontaneity?",
                "Describe your communication style. How do you express yourself?"
            ]
            
            for question in default_questions:
                cursor.execute('INSERT INTO questions (question_text) VALUES (?)', (question,))
            
            conn.commit()
        
        # Create default admin user if none exists
        cursor.execute('SELECT COUNT(*) FROM users WHERE is_admin = TRUE')
        if cursor.fetchone()[0] == 0:
            admin_password = generate_password_hash('admin123')
            cursor.execute('''
                INSERT INTO users (username, email, password_hash, is_admin) 
                VALUES (?, ?, ?, ?)
            ''', ('admin', 'admin@example.com', admin_password, True))
            conn.commit()
            print("Default admin user created: username='admin', password='admin123'")
        
        conn.close()
    
    def create_user(self, username, email, password):
        """Create a new user"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            password_hash = generate_password_hash(password)
            cursor.execute('''
                INSERT INTO users (username, email, password_hash) 
                VALUES (?, ?, ?)
            ''', (username, email, password_hash))
            conn.commit()
            user_id = cursor.lastrowid
            conn.close()
            return user_id
        except sqlite3.IntegrityError:
            conn.close()
            return None
    
    def authenticate_user(self, username, password):
        """Authenticate user login"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT id, password_hash, is_admin FROM users WHERE username = ?', (username,))
        result = cursor.fetchone()
        
        if result and check_password_hash(result[1], password):
            user_id = result[0]
            is_admin = result[2]
            
            # Update last login
            cursor.execute('UPDATE users SET last_login = CURRENT_TIMESTAMP WHERE id = ?', (user_id,))
            conn.commit()
            conn.close()
            
            return {'id': user_id, 'username': username, 'is_admin': bool(is_admin)}
        
        conn.close()
        return None
    
    def get_questions(self):
        """Get all active questions"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT id, question_text FROM questions WHERE is_active = TRUE ORDER BY id')
        questions = cursor.fetchall()
        conn.close()
        
        return [{'id': q[0], 'text': q[1]} for q in questions]
    
    def save_user_responses(self, user_id, responses):
        """Save user responses to questions"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Clear existing responses for this user
        cursor.execute('DELETE FROM user_responses WHERE user_id = ?', (user_id,))
        
        # Insert new responses
        for question_id, response_text in responses.items():
            cursor.execute('''
                INSERT INTO user_responses (user_id, question_id, response_text) 
                VALUES (?, ?, ?)
            ''', (user_id, question_id, response_text))
        
        conn.commit()
        conn.close()
    
    def save_personality_results(self, user_id, predictions, combined_text):
        """Save personality prediction results"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO personality_results 
            (user_id, openness, conscientiousness, extraversion, agreeableness, neuroticism, combined_text) 
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (user_id, predictions['openness'], predictions['conscientiousness'], 
              predictions['extraversion'], predictions['agreeableness'], 
              predictions['neuroticism'], combined_text))
        
        conn.commit()
        result_id = cursor.lastrowid
        conn.close()
        return result_id
    
    def get_user_profile(self, user_id):
        """Get user profile with latest personality results"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get user info
        cursor.execute('SELECT username, email, created_at FROM users WHERE id = ?', (user_id,))
        user_info = cursor.fetchone()
        
        if not user_info:
            conn.close()
            return None
        
        # Get latest personality results
        cursor.execute('''
            SELECT openness, conscientiousness, extraversion, agreeableness, neuroticism, created_at
            FROM personality_results 
            WHERE user_id = ? 
            ORDER BY created_at DESC 
            LIMIT 1
        ''', (user_id,))
        personality_result = cursor.fetchone()
        
        # Get user responses
        cursor.execute('''
            SELECT q.question_text, ur.response_text
            FROM user_responses ur
            JOIN questions q ON ur.question_id = q.id
            WHERE ur.user_id = ?
            ORDER BY q.id
        ''', (user_id,))
        responses = cursor.fetchall()
        
        conn.close()
        
        profile = {
            'username': user_info[0],
            'email': user_info[1],
            'created_at': user_info[2],
            'responses': [{'question': r[0], 'answer': r[1]} for r in responses]
        }
        
        if personality_result:
            profile['personality'] = {
                'openness': personality_result[0],
                'conscientiousness': personality_result[1],
                'extraversion': personality_result[2],
                'agreeableness': personality_result[3],
                'neuroticism': personality_result[4],
                'test_date': personality_result[5]
            }
        
        return profile
    
    def get_all_users(self, include_password=False):
        """Get all users for admin dashboard"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        if include_password:
            cursor.execute('''
                SELECT u.id, u.username, u.email, u.password_hash, u.created_at, u.last_login,
                       COUNT(pr.id) as test_count
                FROM users u
                LEFT JOIN personality_results pr ON u.id = pr.user_id
                WHERE u.is_admin = FALSE
                GROUP BY u.id
                ORDER BY u.created_at DESC
            ''')
            users = cursor.fetchall()
            conn.close()
            return [{'id': u[0], 'username': u[1], 'email': u[2], 'password_hash': u[3],
                    'created_at': u[4], 'last_login': u[5], 'test_count': u[6]} for u in users]
        else:
            cursor.execute('''
                SELECT u.id, u.username, u.email, u.created_at, u.last_login,
                       COUNT(pr.id) as test_count
                FROM users u
                LEFT JOIN personality_results pr ON u.id = pr.user_id
                WHERE u.is_admin = FALSE
                GROUP BY u.id
                ORDER BY u.created_at DESC
            ''')
            users = cursor.fetchall()
            conn.close()
            return [{'id': u[0], 'username': u[1], 'email': u[2],
                    'created_at': u[3], 'last_login': u[4], 'test_count': u[5]} for u in users]

# Flask Application
if FLASK_AVAILABLE:
    app = Flask(__name__)
    app.secret_key = secrets.token_hex(16)
    
    # Initialize components
    db = DatabaseManager()
    predictor = PersonalityPredictor()
    
    @app.route('/')
    def index():
        if 'user_id' in session:
            return redirect(url_for('dashboard'))
        return render_template('index.html')
    
    @app.route('/login', methods=['GET', 'POST'])
    def login():
        if request.method == 'POST':
            username = request.form['username']
            password = request.form['password']
            
            user = db.authenticate_user(username, password)
            if user:
                session['user_id'] = user['id']
                session['username'] = user['username']
                session['is_admin'] = user['is_admin']
                
                if user['is_admin']:
                    return redirect(url_for('admin_dashboard'))
                else:
                    return redirect(url_for('dashboard'))
            else:
                flash('Invalid username or password')
        
        return render_template('login.html')
    
    @app.route('/register', methods=['GET', 'POST'])
    def register():
        if request.method == 'POST':
            username = request.form['username']
            email = request.form['email']
            password = request.form['password']
            
            user_id = db.create_user(username, email, password)
            if user_id:
                flash('Registration successful! Please login.')
                return redirect(url_for('login'))
            else:
                flash('Username or email already exists')
        
        return render_template('register.html')
    
    @app.route('/dashboard')
    def dashboard():
        if 'user_id' not in session:
            return redirect(url_for('login'))
        
        profile = db.get_user_profile(session['user_id'])
        return render_template('dashboard.html', profile=profile)
    
    @app.route('/questionnaire')
    def questionnaire():
        if 'user_id' not in session:
            return redirect(url_for('login'))
        
        questions = db.get_questions()
        return render_template('questionnaire.html', questions=questions)
    
    @app.route('/submit_questionnaire', methods=['POST'])
    def submit_questionnaire():
        if 'user_id' not in session:
            return redirect(url_for('login'))
        
        user_id = session['user_id']
        responses = {}
        combined_text = ""
        
        # Collect responses
        for key, value in request.form.items():
            if key.startswith('question_'):
                question_id = int(key.split('_')[1])
                responses[question_id] = value
                combined_text += value + " "
        
        # Server-side validation
        response_values = list(responses.values())
        
        # Check for minimum length
        for question_id, response in responses.items():
            if len(response.strip()) < 50:
                flash(f'Please provide at least 50 characters for question {question_id}.', 'error')
                return redirect(url_for('questionnaire'))
        
        # Check for identical answers
        if len(response_values) > 1:
            # Convert to lowercase for comparison
            normalized_responses = [resp.strip().lower() for resp in response_values]
            unique_responses = list(set(normalized_responses))
            
            # If more than 70% of answers are identical, reject
            if len(unique_responses) <= len(response_values) * 0.3:
                flash('Please provide more varied and thoughtful answers. Your responses appear too similar across different questions.', 'error')
                return redirect(url_for('questionnaire'))
            
            # Check for very short identical answers
            short_responses = [resp for resp in normalized_responses if len(resp) <= 10]
            if len(short_responses) >= len(response_values) * 0.8:
                flash('Please provide more detailed and thoughtful answers. Short responses don\'t give enough insight into your personality.', 'error')
                return redirect(url_for('questionnaire'))
        
        # Save responses
        db.save_user_responses(user_id, responses)
        
        # Predict personality
        predictions = predictor.predict_personality(combined_text)
        
        # Save results
        db.save_personality_results(user_id, predictions, combined_text)
        
        return redirect(url_for('results'))
    
    @app.route('/results')
    def results():
        if 'user_id' not in session:
            return redirect(url_for('login'))
        
        profile = db.get_user_profile(session['user_id'])
        return render_template('results.html', profile=profile)
    
    @app.route('/admin', methods=['GET', 'POST'])
    def admin_dashboard():
        if 'user_id' not in session or not session.get('is_admin'):
            return redirect(url_for('login'))

        if request.method == 'POST':
            if 'remove_user_id' in request.form:
                # Remove user
                remove_user_id = request.form.get('remove_user_id')
                conn = sqlite3.connect(db.db_path)
                cursor = conn.cursor()
                cursor.execute('DELETE FROM users WHERE id = ?', (remove_user_id,))
                cursor.execute('DELETE FROM user_responses WHERE user_id = ?', (remove_user_id,))
                cursor.execute('DELETE FROM personality_results WHERE user_id = ?', (remove_user_id,))
                conn.commit()
                conn.close()
                flash('User removed successfully (ID: ' + str(remove_user_id) + ')')
                return redirect(url_for('admin_dashboard'))
            else:
                # Handle password change
                user_id = request.form.get('user_id')
                new_password = request.form.get('new_password')
                if user_id and new_password:
                    conn = sqlite3.connect(db.db_path)
                    cursor = conn.cursor()
                    password_hash = generate_password_hash(new_password)
                    cursor.execute('UPDATE users SET password_hash = ? WHERE id = ?', (password_hash, user_id))
                    conn.commit()
                    conn.close()
                    flash('Password updated successfully for user ID ' + str(user_id))
                return redirect(url_for('admin_dashboard'))

        users = db.get_all_users(include_password=True)
        return render_template('admin_dashboard.html', users=users)
    
    @app.route('/admin_login', methods=['GET', 'POST'])
    def admin_login():
        if request.method == 'POST':
            username = request.form['username']
            password = request.form['password']
            user = db.authenticate_user(username, password)
            if user and user['is_admin']:
                session['user_id'] = user['id']
                session['username'] = user['username']
                session['is_admin'] = user['is_admin']
                return redirect(url_for('admin_dashboard'))
            else:
                flash('Invalid admin credentials')
        return render_template('admin_login.html')
    
    @app.route('/logout')
    def logout():
        session.clear()
        return redirect(url_for('index'))

# Demonstration of the system
def demonstrate_system():
    """Demonstrate the personality prediction system"""
    print("=== Personality Prediction Web Application ===\n")
    
    # Initialize components
    print("1. Initializing database...")
    db = DatabaseManager()
    
    print("2. Initializing personality predictor...")
    predictor = PersonalityPredictor()
    
    # Test user creation
    print("\n3. Testing user creation...")
    user_id = db.create_user("testuser", "test@example.com", "password123")
    if user_id:
        print(f"Created test user with ID: {user_id}")
    
    # Test authentication
    print("\n4. Testing authentication...")
    user = db.authenticate_user("testuser", "password123")
    if user:
        print(f"Authentication successful: {user}")
    
    # Test questionnaire
    print("\n5. Testing questionnaire system...")
    questions = db.get_questions()
    print(f"Available questions: {len(questions)}")
    
    # Simulate user responses
    sample_responses = {
        1: "I love exploring new ideas and being creative. Art and music inspire me greatly.",
        2: "I enjoy reading books, visiting museums, and learning about different cultures.",
        3: "When faced with challenges, I try to think creatively and find innovative solutions.",
        4: "I'm motivated by personal growth and making a positive impact on others.",
        5: "I enjoy meeting new people but also value my alone time for reflection."
    }
    
    print("\n6. Saving sample responses...")
    db.save_user_responses(user_id, sample_responses)
    
    # Test personality prediction
    print("\n7. Testing personality prediction...")
    combined_text = " ".join(sample_responses.values())
    predictions = predictor.predict_personality(combined_text)
    
    print("Personality Predictions:")
    for trait, score in predictions.items():
        print(f"  {trait.capitalize()}: {score:.3f}")
    
    # Save results
    result_id = db.save_personality_results(user_id, predictions, combined_text)
    print(f"\nResults saved with ID: {result_id}")
    
    # Test profile retrieval
    print("\n8. Testing profile retrieval...")
    profile = db.get_user_profile(user_id)
    if profile:
        print(f"Profile for {profile['username']}:")
        print(f"  Email: {profile['email']}")
        print(f"  Responses: {len(profile['responses'])}")
        if 'personality' in profile:
            print("  Personality scores:")
            for trait, score in profile['personality'].items():
                if trait != 'test_date':
                    print(f"    {trait.capitalize()}: {score:.3f}")
    
    print("\n=== System demonstration completed! ===")
    
    if FLASK_AVAILABLE:
        print("\nTo run the web application:")
        print("1. Save this code to a file (e.g., app.py)")
        print("2. Create templates folder with HTML files")
        print("3. Run: python app.py")
        print("4. Visit: http://localhost:5000")
        print("\nDefault admin login: username='admin', password='admin123'")
    else:
        print("\nTo run the full web application, install Flask:")
        print("pip install flask")

if __name__ == "__main__":
    if FLASK_AVAILABLE:
        app.run(debug=True)
    else:
        demonstrate_system()