"""
Admin Module
Handles authentication and admin functions
"""

from flask import session, redirect, url_for, request

# Simple user dictionary - no database needed
USERS = {
    'admin': 'admin123',
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


def is_admin():
    """Check if current user is admin"""
    return session.get('username') == 'admin'


def login_user(username, password):
    """Attempt to login user"""
    if username in USERS and USERS[username] == password:
        session['logged_in'] = True
        session['username'] = username
        return True
    return False


def logout_user():
    """Logout current user"""
    session.clear()

