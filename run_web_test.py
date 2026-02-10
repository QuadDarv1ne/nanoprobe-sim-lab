#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple test file to run the web dashboard
"""

from flask import Flask

app = Flask(__name__)

@app.route('/')
def home():
    return '''
    <html>
    <head><title>Nanoprobe Simulation Lab Dashboard</title></head>
    <body>
        <h1>Lab is running!</h1>
        <p>Nanoprobe Simulation Lab Web Dashboard</p>
        <p>Server is running on port 5000</p>
        <p>Status: <span style="color:green;">✓ Active</span></p>
    </body>
    </html>
    '''

if __name__ == '__main__':
    print("Starting web server on http://localhost:5000")
    print("Press Ctrl+C to stop the server")
    app.run(host='127.0.0.1', port=5000, debug=False)