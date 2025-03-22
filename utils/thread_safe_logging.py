"""
Thread-safe logging functions for AI Trading System
"""
import os
import sys
import time
import threading
from datetime import datetime

_log_lock = threading.Lock()

def log_to_file(message, log_file="training_logs.txt"):
    """Thread-safe function to log messages to a file"""
    with _log_lock:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(log_file, "a") as f:
            f.write(f"{timestamp} - {message}\n")
            f.flush()

def log_to_console(message):
    """Thread-safe function to log messages to console"""
    with _log_lock:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"{timestamp} - {message}")
        sys.stdout.flush()

def thread_safe_log(message, log_file="training_logs.txt"):
    """Combined logging function that logs to both file and console"""
    log_to_file(message, log_file)
    log_to_console(message)

def read_logs_from_file(log_file="training_logs.txt", max_lines=100):
    """Read log entries from file with a maximum number of lines"""
    if not os.path.exists(log_file):
        return []
        
    with open(log_file, "r") as f:
        lines = f.readlines()
        
    # Return last N lines (most recent)
    return lines[-max_lines:]