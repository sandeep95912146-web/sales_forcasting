import hashlib
import hmac
import os
import re
import secrets
import sqlite3
import time
from contextlib import closing
from pathlib import Path

# Where the accounts are stored. On a server, set USERS_DB_PATH to a folder that is not erased on restart.
DB_PATH = os.environ.get("USERS_DB_PATH", str(Path(__file__).resolve().parent / "users.db"))
ITERATIONS = 600_000      # slows down password guessing
MAX_FAILED = 5            # wrong passwords before the account is locked
LOCK_SECONDS = 300        # lock time: 5 minutes
COMMON_PASSWORDS = {"password", "password1", "12345678", "123456789", "qwerty123", "iloveyou1", "admin123"}


def _connect():
    con = sqlite3.connect(DB_PATH)
    con.execute(
        """CREATE TABLE IF NOT EXISTS users (
               id INTEGER PRIMARY KEY AUTOINCREMENT,
               username TEXT NOT NULL UNIQUE COLLATE NOCASE,
               email TEXT NOT NULL UNIQUE COLLATE NOCASE,
               salt BLOB NOT NULL,
               password_hash BLOB NOT NULL,
               created_at REAL NOT NULL,
               failed_attempts INTEGER NOT NULL DEFAULT 0,
               locked_until REAL NOT NULL DEFAULT 0,
               last_login REAL)"""
    )
    return con


def _hash(password, salt):
    return hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, ITERATIONS)


def check_username(username):
    if not re.fullmatch(r"[A-Za-z0-9_]{3,30}", username):
        return "Username must be 3 to 30 characters: letters, numbers or underscore."
    return None


def check_email(email):
    if len(email) > 120 or not re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", email):
        return "Please enter a valid email address."
    return None


def check_password(password, username=""):
    if len(password) < 8:
        return "Password must be at least 8 characters."
    if len(password) > 100:
        return "Password is too long."
    if not re.search(r"[A-Za-z]", password) or not re.search(r"\d", password):
        return "Password must contain at least one letter and one number."
    if password.lower() in COMMON_PASSWORDS or (username and password.lower() == username.lower()):
        return "That password is too easy to guess. Please choose another."
    return None


def create_user(username, email, password):
    """Returns (True, message) or (False, reason)."""
    username, email = username.strip(), email.strip()
    problem = check_username(username) or check_email(email) or check_password(password, username)
    if problem:
        return False, problem
    salt = secrets.token_bytes(16)
    try:
        with closing(_connect()) as con, con:
            con.execute(
                "INSERT INTO users (username, email, salt, password_hash, created_at) VALUES (?, ?, ?, ?, ?)",
                (username, email, salt, _hash(password, salt), time.time()),
            )
    except sqlite3.IntegrityError:
        return False, "That username or email is already registered."
    return True, "Account created. Please log in."


def verify_user(username, password):
    """Returns (True, username) or (False, message). The message never says which part was wrong."""
    generic = "Invalid username or password."
    username = username.strip()
    with closing(_connect()) as con, con:
        row = con.execute(
            "SELECT id, username, salt, password_hash, failed_attempts, locked_until FROM users WHERE username = ?",
            (username,),
        ).fetchone()
        if row is None:
            _hash(password, b"0" * 16)          # same waiting time as a real check
            return False, generic
        user_id, real_name, salt, stored_hash, failed, locked_until = row
        now = time.time()
        if locked_until > now:
            minutes = int((locked_until - now) // 60) + 1
            return False, f"Too many wrong attempts. Try again in about {minutes} minute(s)."
        if hmac.compare_digest(_hash(password, salt), stored_hash):
            con.execute("UPDATE users SET failed_attempts = 0, locked_until = 0, last_login = ? WHERE id = ?", (now, user_id))
            return True, real_name
        failed += 1
        if failed >= MAX_FAILED:
            con.execute("UPDATE users SET failed_attempts = 0, locked_until = ? WHERE id = ?", (now + LOCK_SECONDS, user_id))
            return False, "Too many wrong attempts. This account is locked for 5 minutes."
        con.execute("UPDATE users SET failed_attempts = ? WHERE id = ?", (failed, user_id))
        return False, generic
