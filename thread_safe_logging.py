"""
Thread-safe logging functions for AI Trading System
"""
import threading
import datetime
import os

# Thread-safe lock for logging
log_lock = threading.Lock()

def log_to_file(message, log_file="training_logs.txt"):
    """Thread-safe function to log messages to a file"""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"{timestamp} - {message}\n"
    
    with log_lock:
        try:
            with open(log_file, "a") as f:
                f.write(log_entry)
        except Exception as e:
            print(f"Error writing to log file: {e}")

def log_to_console(message):
    """Thread-safe function to log messages to console"""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"{timestamp} - {message}"
    
    with log_lock:
        print(log_entry)

def thread_safe_log(message, log_file="training_logs.txt"):
    """Combined logging function that logs to both file and console"""
    log_to_file(message, log_file)
    log_to_console(message)

def read_logs_from_file(log_file="training_logs.txt", max_lines=100):
    """Read log entries from file with a maximum number of lines"""
    if not os.path.exists(log_file):
        return []
        
    try:
        with open(log_file, "r") as f:
            # Read last max_lines lines
            lines = f.readlines()
            return lines[-max_lines:] if len(lines) > max_lines else lines
    except Exception as e:
        print(f"Error reading log file: {e}")
        return []