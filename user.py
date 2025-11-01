"""
User Interface
Login required - View reports and entries (no graphs)
"""

from flask import Flask, render_template, jsonify, request, session, redirect
from storage import DataStorage

app = Flask(__name__)
app.secret_key = 'user-secret-key-2024'

storage = DataStorage()

# User credentials
USER_ACCOUNTS = {
    'user': 'user123',
    'viewer': 'viewer123'
}


def require_login(f):
    """Decorator to require login"""
    def decorated_function(*args, **kwargs):
        if not session.get('logged_in'):
            return redirect('/login')
        return f(*args, **kwargs)
    decorated_function.__name__ = f.__name__
    return decorated_function


@app.route('/login', methods=['GET', 'POST'])
def login():
    """User login page"""
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        if username in USER_ACCOUNTS and USER_ACCOUNTS[username] == password:
            session['logged_in'] = True
            session['username'] = username
            return redirect('/')
        else:
            return render_template('userlogin.html', error='Invalid credentials')
    
    return render_template('userlogin.html')


@app.route('/logout')
def logout():
    """Logout"""
    session.clear()
    return redirect('/login')


@app.route('/')
@require_login
def index():
    """User dashboard"""
    return render_template('user.html')


@app.route('/api/entries')
@require_login
def get_entries():
    """Get all entries"""
    all_data = storage.get_all_detections()
    entries = []
    
    for det in reversed(all_data):  # Latest first
        entries.append({
            'date': det.get('date', ''),
            'time': det.get('time', ''),
            'datetime': det.get('datetime', ''),
            'compliant': det.get('is_compliant', False),
            'compliance_percentage': det.get('compliance_percentage', 0),
            'missing_items': [item.get('name', '') for item in det.get('missing_items', [])],
            'present_items': [item.get('name', '') for item in det.get('present_items', [])]
        })
    
    return jsonify(entries)


@app.route('/api/stats')
@require_login
def get_stats():
    """Get basic statistics"""
    all_data = storage.get_all_detections()
    
    total = len(all_data)
    violations = len([d for d in all_data if not d.get('is_compliant', False)])
    compliant = total - violations
    compliance_rate = (compliant / total * 100) if total > 0 else 0
    
    today = storage.get_today_detections()
    today_violations = len([d for d in today if not d.get('is_compliant', False)])
    
    return jsonify({
        'total': total,
        'violations': violations,
        'compliant': compliant,
        'compliance_rate': round(compliance_rate, 2),
        'today_total': len(today),
        'today_violations': today_violations
    })


if __name__ == '__main__':
    print("Starting User Interface...")
    print("Open your browser and go to: http://localhost:5002")
    print("Login: user/user123 or viewer/viewer123")
    app.run(debug=False, host='0.0.0.0', port=5002)

