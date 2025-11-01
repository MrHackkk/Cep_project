"""
Admin Interface
Login required - View reports with graphs and statistics
"""

from flask import Flask, render_template, jsonify, request, session, redirect
from storage import DataStorage
from datetime import datetime, timedelta
from collections import defaultdict

app = Flask(__name__)
app.secret_key = 'admin-secret-key-2024'

storage = DataStorage()

# Admin credentials
ADMIN_USERS = {
    'admin': 'admin123'
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
    """Admin login page"""
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        if username in ADMIN_USERS and ADMIN_USERS[username] == password:
            session['logged_in'] = True
            session['username'] = username
            return redirect('/')
        else:
            return render_template('adminlogin.html', error='Invalid credentials')
    
    return render_template('adminlogin.html')


@app.route('/logout')
def logout():
    """Logout"""
    session.clear()
    return redirect('/login')


@app.route('/')
@require_login
def index():
    """Admin dashboard"""
    return render_template('admin.html')


@app.route('/api/stats')
@require_login
def get_stats():
    """Get statistics for graphs"""
    all_data = storage.get_all_detections()
    
    # Daily statistics
    daily_stats = defaultdict(lambda: {'total': 0, 'violations': 0, 'compliant': 0})
    for detection in all_data:
        date = detection.get('date', '')
        if date:
            daily_stats[date]['total'] += 1
            if not detection.get('is_compliant', False):
                daily_stats[date]['violations'] += 1
            else:
                daily_stats[date]['compliant'] += 1
    
    # Last 7 days
    last_7_days = []
    for i in range(6, -1, -1):
        date = (datetime.now() - timedelta(days=i)).strftime('%Y-%m-%d')
        stats = daily_stats.get(date, {'total': 0, 'violations': 0, 'compliant': 0})
        last_7_days.append({
            'date': date,
            'total': stats['total'],
            'violations': stats['violations'],
            'compliant': stats['compliant']
        })
    
    # Overall statistics
    total = len(all_data)
    violations = len([d for d in all_data if not d.get('is_compliant', False)])
    compliant = total - violations
    compliance_rate = (compliant / total * 100) if total > 0 else 0
    
    # Missing items count
    missing_items_count = defaultdict(int)
    for detection in all_data:
        for item in detection.get('missing_items', []):
            missing_items_count[item.get('name', 'Unknown')] += 1
    
    # Daily detection count (how many users detected regularly)
    detection_dates = defaultdict(int)
    for detection in all_data:
        date = detection.get('date', '')
        if date:
            detection_dates[date] += 1
    
    regular_days = len([count for count in detection_dates.values() if count >= 10])
    
    return jsonify({
        'overall': {
            'total': total,
            'violations': violations,
            'compliant': compliant,
            'compliance_rate': round(compliance_rate, 2)
        },
        'daily_stats': last_7_days,
        'missing_items': dict(missing_items_count),
        'regular_users': regular_days,
        'total_days': len(detection_dates)
    })


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


if __name__ == '__main__':
    print("Starting Admin Interface...")
    print("Open your browser and go to: http://localhost:5001")
    print("Login: admin / admin123")
    app.run(debug=False, host='0.0.0.0', port=5001)
