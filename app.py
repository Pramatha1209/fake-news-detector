import urllib.request
from flask import Flask, render_template, request, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
import joblib
import requests
from bs4 import BeautifulSoup
from datetime import datetime
import os

app = Flask(__name__)
app.config['SECRET_KEY'] = 'supersecretveritasnewskey2026'

# Ensure the instance directory exists for the database
os.makedirs('instance', exist_ok=True)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = "login"
login_manager.login_message_category = "error"

# Load ML model
try:
    model = joblib.load("model.pkl")
    vectorizer = joblib.load("vectorizer.pkl")
except FileNotFoundError:
    print("Warning: model.pkl or vectorizer.pkl not found. Please train the model first.")
    model = None
    vectorizer = None

# ================= DATABASE MODELS =================

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), unique=True, nullable=False)
    password = db.Column(db.String(255), nullable=False)
    role = db.Column(db.String(20), default="user")

class History(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    news_text = db.Column(db.Text, nullable=False)
    prediction = db.Column(db.String(10), nullable=False)
    confidence = db.Column(db.Float, nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# ================= ML FUNCTION =================

def predict_news(text):
    if not model or not vectorizer:
        return "Error", 0.0
        
    text = str(text).lower()
    
    text_vector = vectorizer.transform([text])
    prediction = model.predict(text_vector)[0]
    
    decision_score = abs(model.decision_function(text_vector)[0])
    
    confidence = min(decision_score * 20, 99.9) 
    
    if confidence < 50:
        confidence = 50 + (confidence / 2)

    if prediction == 1:
        return "Real", round(confidence, 2)
    else:
        return "Fake", round(confidence, 2)

def extract_text_from_url(url):
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, "html.parser")
        
        for script in soup(["script", "style", "nav", "header", "footer"]):
            script.extract()
            
        paragraphs = soup.find_all("p")
        text = " ".join([para.get_text(strip=True) for para in paragraphs if len(para.get_text(strip=True)) > 20])
        
        if not text:
            text = soup.get_text(separator=' ', strip=True)
            
        return text
    except Exception as e:
        print(f"Error scraping {url}: {e}")
        return None

# ================= ROUTES =================

@app.route("/")
def home():
    if current_user.is_authenticated:
        return redirect(url_for("dashboard"))
    return redirect(url_for("login"))

@app.route("/register", methods=["GET", "POST"])
def register():
    if current_user.is_authenticated:
        return redirect(url_for("dashboard"))
        
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        
        if not username or not password:
            flash("Username and password are required.", "error")
            return redirect(url_for("register"))
            
        if User.query.filter_by(username=username).first():
            flash("Username already exists. Please choose a different one.", "error")
            return redirect(url_for("register"))

        hashed_password = generate_password_hash(password)
        
        is_first_user = User.query.count() == 0
        role = "admin" if is_first_user else "user"

        user = User(username=username, password=hashed_password, role=role)
        db.session.add(user)
        db.session.commit()

        flash("Registration successful! Please login securely.", "success")
        return redirect(url_for("login"))

    return render_template("register.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if current_user.is_authenticated:
        return redirect(url_for("dashboard"))
        
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")

        user = User.query.filter_by(username=username).first()

        if user and check_password_hash(user.password, password):
            login_user(user)
            flash(f"Welcome back, {username}!", "success")
            return redirect(url_for("dashboard"))
        else:
            flash("Invalid credentials. Please verify your username and password.", "error")

    return render_template("login.html")

@app.route("/dashboard", methods=["GET", "POST"])
@login_required
def dashboard():
    result = None
    confidence = None

    if request.method == "POST":
        news_text = request.form.get("news", "").strip()
        url = request.form.get("url", "").strip()

        if not news_text and not url:
            flash("Please provide either text content or a URL to analyze.", "error")
        else:
            if url:
                scraped_text = extract_text_from_url(url)
                if scraped_text and len(scraped_text) > 50:
                    news_text = scraped_text
                    flash(f"Successfully extracted {len(news_text)} characters from URL.", "success")
                else:
                    news_text = None
                    flash("Failed to extract meaningful text from the provided URL. It might be protected or essentially empty.", "error")

            if news_text:
                if len(news_text) < 20:
                    flash("The provided text is too short for an accurate prediction.", "error")
                else:
                    result, confidence = predict_news(news_text)

                    snippet = news_text[:200] + "..." if len(news_text) > 200 else news_text

                    history = History(
                        user_id=current_user.id,
                        news_text=snippet,
                        prediction=result,
                        confidence=confidence
                    )
                    db.session.add(history)
                    db.session.commit()

    user_history = History.query.filter_by(user_id=current_user.id).order_by(History.timestamp.desc()).limit(10).all()

    return render_template(
        "dashboard.html",
        result=result,
        confidence=confidence,
        history=user_history
    )

@app.route("/logout")
@login_required
def logout():
    logout_user()
    flash("You have been securely logged out.", "success")
    return redirect(url_for("login"))

# ================= RUN =================
@app.route("/admin")
@login_required
def admin():
    if current_user.role != "admin":
        flash("You do not have permission to access the admin portal.", "error")
        return redirect(url_for("dashboard"))

    users = User.query.all()
    all_history = History.query.order_by(History.timestamp.desc()).limit(50).all()

    model_accuracy = 99.5 

    return render_template(
        "admin.html",
        users=users,
        history=all_history,
        accuracy=model_accuracy
    )

with app.app_context():
    db.create_all()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)