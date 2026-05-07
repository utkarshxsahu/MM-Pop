"""
Logging utilities
"""

import os
from datetime import datetime

class Logger:
    """Simple logger that writes to both console and file"""
    def __init__(self, log_file):
        self.log_file = log_file
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
        # Clear log file
        with open(self.log_file, 'w') as f:
            f.write(f"Log started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*80 + "\n\n")
    
    def log(self, message, console=True):
        """Write message to log file and optionally console"""
        if console:
            print(message)
        
        with open(self.log_file, 'a') as f:
            f.write(message + '\n')
    
    def section(self, title, console=True):
        """Log a section header"""
        separator = "="*80
        self.log(f"\n{separator}", console)
        self.log(title, console)
        self.log(separator, console)
