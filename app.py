import os
from flask import Flask, request, redirect, url_for, send_file, flash, send_from_directory, render_template, jsonify
from werkzeug.utils import secure_filename
from PyPDF2 import PdfReader, PdfWriter
from reportlab.pdfgen import canvas
from io import BytesIO

import fitz  # PyMuPDF
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
import tqdm
import matplotlib
matplotlib.use('Agg')

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['PROCESSED_FOLDER'] = 'processed'
app.secret_key = 'your_secret_key'  # Replace with your actual secret key

ALLOWED_EXTENSIONS = {'pdf'}

def extract_data_from_pdf(pdf_path):
    # Open the PDF document
    doc = fitz.open(pdf_path)

    # Initialize a list to hold the rows
    rows = []

    # Iterate through the pages
    for page_num in range(doc.page_count):
        page = doc.load_page(page_num)
        text = page.get_text("text")
        lines = text.strip().split('\n')
        
        # Filter out lines that contain specific words or letters
        filtered_lines = [line for line in lines if not re.search(r'\b(Centre|Page No|2024)\b', line)]
        filtered_lines = [line.strip() for line in filtered_lines if not re.search(r'[a-zA-Z]', line.strip())]
        
        filtered_text = '\n'.join(filtered_lines)
        numbers = re.findall(r'\b\d+\b', filtered_text)
        numbers = [int(num) for num in numbers]
        
        # Ensure the number of items is even
        if len(numbers) % 2 != 0:
            raise ValueError("Number of elements is not even, cannot form pairs.")
        
        pairs = [(numbers[i], numbers[i+1]) for i in range(0, len(numbers) - 1, 2)]

        # Process pairs and append to rows
        for serial, marks in pairs:
            if marks == 2024:
                continue  # Skip this pair
            rows.append((serial, marks))
    
    # Convert the rows into a pandas DataFrame
    df = pd.DataFrame(rows, columns=['Serial Number', 'Marks'])

    return df

def calculate_unobtainable_scores():
    max_questions = 180
    possible_scores = set()
    
    for correct in range(max_questions + 1):
        for no_answer in range(max_questions - correct + 1):
            score = 5 * correct - max_questions + no_answer
            possible_scores.add(score)
    
    return set([717, 718, 719])

def analyze_and_save_to_pdf(df, pdf_filename):
    with PdfPages(pdf_filename) as pdf:
        # Descriptive statistics
        
        # Descriptive statistics
        mean = round(df['Marks'].mean(), 2)
        median = df['Marks'].median()
        mode = df['Marks'].mode()[0]
        min_marks = df['Marks'].min()
        max_marks = df['Marks'].max()
        percentiles = np.percentile(df['Marks'], [10, 25, 50, 75, 90])
        
        # Summary Table
        summary_data = {
            'Statistic': ['Mean', 'Median', 'Mode', 'Minimum', 'Maximum', '10th Percentile', '25th Percentile', '50th Percentile', '75th Percentile', '90th Percentile'],
            'Value': [mean, median, mode, min_marks, max_marks, percentiles[0], percentiles[1], percentiles[2], percentiles[3], percentiles[4]]
        }
        summary_df = pd.DataFrame(summary_data)
        
        # Modern Table Plot
        plt.figure(figsize=(12, 4))
        plt.subplot(111, frame_on=False)
        table = plt.table(cellText=summary_df.values,
                        colLabels=summary_df.columns,
                        cellLoc='center',
                        loc='center',
                        bbox=[0, 0, 1, 1])
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1.2, 1.2)
        plt.axis('off')
        plt.title('Summary Statistics', pad=12, fontsize=16, fontweight='bold')
        pdf.savefig()
        plt.close()

        # Frequency Table of Specific Marks
        highest_marks = df['Marks'].max()
        lowest_marks = df['Marks'].min()
        most_frequent_marks = df['Marks'].mode()[0]
        
        freq_specific_marks = df['Marks'].value_counts().loc[[highest_marks, lowest_marks, most_frequent_marks]].reset_index()
        freq_specific_marks.columns = ['Marks', 'Frequency']
        
        plt.figure(figsize=(10, 4))
        plt.subplot(111, frame_on=False)
        table = plt.table(cellText=freq_specific_marks.values,
                        colLabels=freq_specific_marks.columns,
                        cellLoc='center',
                        loc='center',
                        bbox=[0, 0, 1, 1])
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1.2, 1.2)
        plt.axis('off')
        plt.title('Frequency of Highest, Lowest, and Most Frequent Marks', pad=12, fontsize=16, fontweight='bold')
        pdf.savefig()
        plt.close()

        # Histogram
        plt.figure(figsize=(10, 6))
        plt.hist(df['Marks'], bins=60, edgecolor='dodgerblue', color='lightblue')
        plt.title('Histogram of Marks', fontsize=16, fontweight='bold')
        plt.xlabel('Marks', fontsize=14)
        plt.ylabel('Frequency', fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.7)
        pdf.savefig()
        plt.close()
        
        # Density Plot
        plt.figure(figsize=(10, 6))
        sns.kdeplot(df['Marks'], shade=True, color='darkorange', linewidth=2)
        plt.title('Density Plot of Marks', fontsize=16, fontweight='bold')
        plt.xlabel('Marks', fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.7)
        pdf.savefig()
        plt.close()

        # Outliers Table
        df['Z-Score'] = (df['Marks'] - mean) / df['Marks'].std()
        outliers = df[np.abs(df['Z-Score']) > 2]
        
        plt.figure(figsize=(10, 6))
        plt.subplot(111, frame_on=False)
        table = plt.table(cellText=outliers[['Serial Number', 'Marks']].values,
                        colLabels=['Serial Number', 'Marks'],
                        cellLoc='center',
                        loc='center',
                        bbox=[0, 0, 1, 1])
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1.2, 1.2)
        plt.axis('off')
        plt.title('Outliers Table', pad=12, fontsize=16, fontweight='bold')
        pdf.savefig()
        plt.close()
        
        # Box Plot with Outliers Highlighted
        plt.figure(figsize=(10, 6))
        sns.set(style='whitegrid')
        sns.boxplot(x=df['Marks'], color='skyblue')
        plt.scatter(outliers['Marks'], np.ones(len(outliers)), color='red', label='Outliers', zorder=5)
        plt.title('Box Plot of Marks with Outliers Highlighted', fontsize=16, fontweight='bold')
        plt.xlabel('Marks', fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        pdf.savefig()
        plt.close()

    print(f"PDF saved as {pdf_filename}")

def generate_summary(input_pdf_file, output_pdf_file):
    return analyze_and_save_to_pdf(extract_data_from_pdf(input_pdf_file), output_pdf_file)

@app.route('/')
def upload():
    return render_template('index.html')

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def process_pdf(input_pdf_path):
    centre_code = input_pdf_path.split('/')[-1].split('.')[0]
    output_name = f"{centre_code}_centre_analysis_neetug24.pdf"
    output_pdf_path = os.path.join(app.config['PROCESSED_FOLDER'], output_name)
    print(output_pdf_path)
    # Example PDF processing (you can replace this with your actual processing logic)
    with open(input_pdf_path, 'rb') as input_pdf_file:
        reader = PdfReader(input_pdf_file)
        generate_summary(input_pdf_file=input_pdf_path, output_pdf_file=output_pdf_path)

    return output_pdf_path

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Process the PDF
        processed_pdf_path = process_pdf(file_path)
        processed_pdf_filename = os.path.basename(processed_pdf_path)

        flash('File successfully uploaded and processed')
        # Create URL to directly open the PDF
        pdf_url = url_for('download_pdf', filename=processed_pdf_filename)
        return jsonify({'url': pdf_url})  # Return URL in JSON response
    else:
        flash('Allowed file types are pdf')
        return redirect(request.url)
  
@app.route('/download_pdf')
def download_pdf():
    filename = request.args.get('filename')
    if filename:
        # Ensure filename is safe to use
        safe_filename = secure_filename(filename)
        file_path = os.path.join(app.config['PROCESSED_FOLDER'], safe_filename)
        
        if os.path.isfile(file_path):
            print("found file:", file_path)
            # Use send_file to serve the file
            return send_file(file_path, as_attachment=True)
        else:
            flash('File not found')
            return redirect('/')
    else:
        flash('No filename provided')
        return redirect('/')

if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    if not os.path.exists(app.config['PROCESSED_FOLDER']):
        os.makedirs(app.config['PROCESSED_FOLDER'])
    app.run(debug=True)
