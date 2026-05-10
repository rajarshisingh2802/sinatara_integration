from flask import Flask, render_template, request, Response, redirect, url_for, session, flash, jsonify
from werkzeug.security import generate_password_hash, check_password_hash
import sqlite3
import json
import random 
import smtplib 
from email.message import EmailMessage
import razorpay
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import os

# --- NEW: Import your AI Pipeline ---
# Make sure the file is saved as ai_pipeline.py in the same directory
import ai_pipeline

app = Flask(__name__)
app.secret_key = 'super_secret_promiq_key_change_this_later' 
DB_NAME = 'promiq.db'

# --- Database Setup ---
def init_db():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            credits INTEGER DEFAULT 10 
        )
    ''')
    
    c.execute('''
        CREATE TABLE IF NOT EXISTS prompt_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            raw_prompt TEXT,
            refined_prompt TEXT,
            final_output TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')
    conn.commit()
    conn.close()

def get_db_connection():
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    return conn

init_db()

# --- Helper Function for OTP ---
def send_otp(email, otp):
    SMTP_SERVER = "smtp.gmail.com"
    SMTP_PORT = 587
    EMAIL_USER = os.getenv("EMAIL_USER")
    EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")
    SENDER_EMAIL = EMAIL_USER
    SENDER_PASSWORD = EMAIL_PASSWORD 

    html_body = f"""
    <html>
        <body style="font-family: Arial, sans-serif; color: #333; line-height: 1.6; padding: 20px;">
            <div style="max-width: 500px; margin: auto; border: 1px solid #e5e7eb; border-radius: 10px; padding: 30px; text-align: center;">
                <h2 style="color: #8b5cf6;">Sintara</h2>
                <p>Hello,</p>
                <p>Your one-time verification code is:</p>
                <h1 style="font-size: 36px; letter-spacing: 5px; color: #111827; background: #f3f4f6; padding: 10px; border-radius: 8px;">{otp}</h1>
                <p style="font-size: 12px; color: #9ca3af;">This code will expire in 10 minutes.</p>
            </div>
        </body>
    </html>
    """
    
    msg = MIMEMultipart()
    msg['Subject'] = 'Your OTP'
    msg['From'] = f"Sintara Support <{SENDER_EMAIL}>"
    msg['To'] = email
    msg.attach(MIMEText(html_body, 'html'))

    try:
        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
        server.starttls() 
        server.login(SENDER_EMAIL, SENDER_PASSWORD)
        server.send_message(msg)
        server.quit()
        print("✅ Email sent successfully to", email)
        return True
    except Exception as e:
        print("❌ Email failed to send:", e)
        return False


# --- Application Routes ---
@app.route('/')
def dashboard():
    user_name = session.get('user_name', None)
    credits_left = 2

    if 'user_id' in session:
        conn = get_db_connection()
        user = conn.execute('SELECT credits FROM users WHERE id = ?', (session['user_id'],)).fetchone()
        conn.close()
        if user:
            credits_left = user['credits']
    else:
        credits_left = session.get('anon_credits', 2)
        if 'anon_credits' not in session:
            session['anon_credits'] = credits_left

    return render_template('dashboard.html', user_name=user_name, credits_left=credits_left)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')

        conn = get_db_connection()
        user = conn.execute('SELECT * FROM users WHERE email = ?', (email,)).fetchone()
        conn.close()

        if user and check_password_hash(user['password'], password):
            session['user_id'] = user['id']
            session['user_name'] = user['name']
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid email or password. Please try again.')
            return redirect(url_for('login'))
            
    return render_template('login.html', active_form='login')

@app.route('/signup', methods=['POST'])
def signup():
    name = request.form.get('name')
    email = request.form.get('email')
    password = request.form.get('password')

    conn = get_db_connection()
    existing_user = conn.execute('SELECT id FROM users WHERE email = ?', (email,)).fetchone()
    conn.close()

    if existing_user:
        flash('This email is already registered! Please log in instead.')
        return redirect(url_for('login'))

    otp = str(random.randint(100000, 999999))
    session['temp_signup'] = {'name': name, 'email': email, 'password': password, 'otp': otp}
    
    send_otp(email, otp)
    flash('An OTP has been sent to your email. (Check your terminal!)')
    return render_template('login.html', active_form='verify_signup')

@app.route('/verify-signup', methods=['POST'])
def verify_signup():
    entered_otp = request.form.get('otp')
    temp_data = session.get('temp_signup')

    if not temp_data or temp_data['otp'] != entered_otp:
        flash('Invalid or expired OTP. Please try signing up again.')
        return render_template('login.html', active_form='signup')

    hashed_password = generate_password_hash(temp_data['password'], method='pbkdf2:sha256')
    
    conn = get_db_connection()
    conn.execute('INSERT INTO users (name, email, password) VALUES (?, ?, ?)', 
                 (temp_data['name'], temp_data['email'], hashed_password))
    conn.commit()
    conn.close()
    
    session.pop('temp_signup', None) 
    flash('Account verified and created! Please log in.')
    return redirect(url_for('login'))

@app.route('/forgot-password', methods=['GET', 'POST'])
def forgot_password():
    if request.method == 'POST':
        email = request.form.get('email')
        
        conn = get_db_connection()
        user = conn.execute('SELECT id FROM users WHERE email = ?', (email,)).fetchone()
        conn.close()

        if user:
            otp = str(random.randint(100000, 999999))
            session['temp_reset'] = {'email': email, 'otp': otp}
            send_otp(email, otp)
            flash('A reset code has been sent to your email. (Check terminal!)')
            return render_template('login.html', active_form='reset_password')
        else:
            flash('No account found with that email.')
            return redirect(url_for('forgot_password'))
            
    return render_template('login.html', active_form='forgot_password')

@app.route('/reset-password', methods=['POST'])
def reset_password():
    entered_otp = request.form.get('otp')
    new_password = request.form.get('new_password')
    temp_data = session.get('temp_reset')

    if not temp_data or temp_data['otp'] != entered_otp:
        flash('Invalid OTP. Please request a new password reset.')
        return render_template('login.html', active_form='forgot_password')

    hashed_password = generate_password_hash(new_password, method='pbkdf2:sha256')
    
    conn = get_db_connection()
    conn.execute('UPDATE users SET password = ? WHERE email = ?', 
                 (hashed_password, temp_data['email']))
    conn.commit()
    conn.close()
    
    session.pop('temp_reset', None)
    flash('Password successfully updated! You can now log in.')
    return redirect(url_for('login'))

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('dashboard'))

@app.route('/terms')
def terms():
    return render_template('terms.html')

razorpay_client = razorpay.Client(auth=("rzp_test_SkUtVi38fMkUOU", "JqlXgCWz0XFDm2wM07hJYg2l"))

@app.route('/create-razorpay-order', methods=['POST'])
def create_order():
    if 'user_id' not in session:
        return jsonify({'error': 'Please log in to buy credits.'}), 401

    try:
        order_data = {
            "amount": 9900, 
            "currency": "INR",
            "receipt": f"receipt_{session['user_id']}",
            "notes": {
                "user_id": session['user_id'],
                "product": "50 Sintara Credits"
            }
        }
        order = razorpay_client.order.create(data=order_data)
        return jsonify({'order_id': order['id'], 'amount': order['amount']})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/verify-payment', methods=['POST'])
def verify_payment():
    data = request.get_json()
    
    try:
        params_dict = {
            'razorpay_order_id': data['razorpay_order_id'],
            'razorpay_payment_id': data['razorpay_payment_id'],
            'razorpay_signature': data['razorpay_signature']
        }
        razorpay_client.utility.verify_payment_signature(params_dict)
        
        conn = get_db_connection()
        conn.execute('UPDATE users SET credits = credits + 50 WHERE id = ?', (session['user_id'],))
        conn.commit()
        
        user = conn.execute('SELECT credits FROM users WHERE id = ?', (session['user_id'],)).fetchone()
        new_credits = user['credits']
        conn.close()
        
        return jsonify({'status': 'success', 'new_credits': new_credits})
        
    except razorpay.errors.SignatureVerificationError:
        return jsonify({'status': 'failed', 'error': 'Invalid payment signature'}), 400
    except Exception as e:
        return jsonify({'status': 'failed', 'error': str(e)}), 500

# =====================================================================
# --- AI Core Route: Execute Pipeline to Refine Prompt ---
# =====================================================================
@app.route('/refine', methods=['POST'])
def refine_prompt():
    if 'user_id' in session:
        conn = get_db_connection()
        user = conn.execute('SELECT credits FROM users WHERE id = ?', (session['user_id'],)).fetchone()
        
        if not user or user['credits'] <= 0:
            conn.close()
            return jsonify({'error': 'You have 0 credits left! Please upgrade.'}), 403
            
        conn.execute('UPDATE users SET credits = credits - 1 WHERE id = ?', (session['user_id'],))
        conn.commit()
        credits_left = user['credits'] - 1
        conn.close()
    else:
        anon_credits = session.get('anon_credits', 2)
        if anon_credits <= 0:
            return jsonify({'error': 'Guest limit reached. Please sign up for 10 free credits!'}), 403
        
        session['anon_credits'] = anon_credits - 1
        credits_left = session['anon_credits']

    data = request.get_json()
    user_prompt = data.get('prompt')

    if not user_prompt:
        return jsonify({'error': 'No prompt provided'}), 400

    try:
        # Pass the input directly into the imported pipeline logic
        result = ai_pipeline.run_full_pipeline(intent=user_prompt)
        
        # Save history
        user_id = session.get('user_id')
        conn = get_db_connection()
        conn.execute('''
            INSERT INTO prompt_history (user_id, raw_prompt, refined_prompt) 
            VALUES (?, ?, ?)
        ''', (user_id, user_prompt, result['final_winner']))
        conn.commit()
        conn.close()
        
        # Respond with strictly what the pipeline outputs
        return jsonify({
            'refined_prompt': result['final_winner'],
            'credits_left': credits_left
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)