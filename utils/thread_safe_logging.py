"""
Thread-safe logging functions for AI Trading System
"""
import os
import time
import threading

_log_lock = threading.Lock()

def log_to_file(message, log_file="training_logs.txt"):
    """Thread-safe function to log messages to a file"""
    with _log_lock:
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        try:
            with open(log_file, "a") as f:
                f.write(f"{timestamp} - {message}\n")
        except Exception as e:
            print(f"Error writing to log file: {str(e)}")

def log_to_console(message):
    """Thread-safe function to log messages to console"""
    with _log_lock:
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        print(f"{timestamp} - {message}")

def thread_safe_log(message, log_file="training_logs.txt"):
    """Combined logging function that logs to both file and console"""
    log_to_file(message, log_file)
    log_to_console(message)

def read_logs_from_file(log_file="training_logs.txt", max_lines=100):
    """Read log entries from file with a maximum number of lines"""
    try:
        if not os.path.exists(log_file):
            return ["No log file found"]
            
        with open(log_file, "r") as f:
            lines = f.readlines()
            
        return lines[-max_lines:] if len(lines) > max_lines else lines
    except Exception as e:
        return [f"Error reading log file: {str(e)}"]