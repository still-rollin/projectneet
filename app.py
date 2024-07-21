import os
from flask import Flask, request, redirect, url_for, send_file, flash, send_from_directory, render_template, jsonify
from werkzeug.utils import secure_filename
from PyPDF2 import PdfReader, PdfWriter
from reportlab.pdfgen import canvas
from io import BytesIO
import textwrap

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


def get_centre_info(text):
     # Split text into lines
    lines = text.strip().split('\n')
    
    # Initialize variables
    centre_name = ''
    centre_id = None
    centre_line_index = None
    
    # Find the "Centre" line and its index
    for i, line in enumerate(lines):
        if 'Centre' in line:
            centre_line_index = i
            # Extract Centre ID
            match = re.search(r'Centre\s*:\s*(\d+)', line)
            if match:
                centre_id = match.group(1)
            break
    
    # If the Centre line was found, find the longest line after it
    if centre_line_index is not None and centre_id:
        # Get lines after the "Centre" line
        lines_after_centre = lines[centre_line_index + 1:]
        # Find the longest line
        centre_name = max(lines_after_centre, key=lambda x: len(x.strip()), default='').strip()
        return {'name':centre_name , 'id':centre_id }
    else:
        return 'Centre information not found'
    

def extract_data_from_pdf(pdf_path):
    # Open the PDF document
    doc = fitz.open(pdf_path)

    # Initialize a list to hold the rows
    rows = []

    # Iterate through the pages
    for page_num in range(doc.page_count):
        page = doc.load_page(page_num)
        text = page.get_text("text")
        centre_info = get_centre_info(text)
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

    return df, centre_info


def calculate_unobtainable_scores():
    max_questions = 180
    possible_scores = set()
    
    for correct in range(max_questions + 1):
        for no_answer in range(max_questions - correct + 1):
            score = 5 * correct - max_questions + no_answer
            possible_scores.add(score)
    
    return set([717, 718, 719])

disclaimer_text = (
    "Disclaimer: The information presented in this document is for informational purposes only.\n "
    "We do not have any legal obligation for the accuracy of these results and this document is not legally binding."
)
 
def add_disclaimer(fig):
    """Add disclaimer text to the bottom of the figure."""
    fig.text(0.5, 0.02, 
            disclaimer_text,
            ha='center', va='center', fontsize=8, fontweight='light', color='grey',
            bbox=dict(facecolor='lightgrey', alpha=0.5, boxstyle='round,pad=0.5'))

def analyze_and_save_to_pdf(df, centre_info, pdf_filename):
    # Disclaimer Text - Wrapped
   
    with PdfPages(pdf_filename) as pdf:
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.text(0.5, 0.8, 'NEET UG 2024 Centre Score Distribution Analysis', 
                fontsize=20, fontweight='bold', ha='center', va='center')

        centre_name = centre_info["name"]
        wrapped_name = textwrap.fill(f'{centre_name}', width=50)
        wrapped_id = f'Centre ID: {centre_info["id"]}'

        # Display wrapped text with styled boxes and adjusted positions
        ax.text(0.5, 0.35, wrapped_id, fontsize=16, ha='center', va='center',
                bbox=dict(facecolor='lightgrey', edgecolor='black', boxstyle='round,pad=0.5'))
        ax.text(0.5, 0.60, wrapped_name, fontsize=16, ha='center', va='center',
                bbox=dict(facecolor='lightblue', edgecolor='black', boxstyle='round,pad=0.5'))

        ax.axis('off')

        # Save the page
        pdf.savefig(fig)
        plt.close(fig)
        
        # Summary Table
        mean = round(df['Marks'].mean(), 2)
        median = round(df['Marks'].median(), 2)
        mode = df['Marks'].mode()[0]
        min_marks = df['Marks'].min()
        max_marks = df['Marks'].max()
        percentiles = np.percentile(df['Marks'], [10, 25, 50, 75, 90])
        percentiles = [round(i) for i in percentiles]
        
        summary_data = {
            'Statistic': ['Mean', 'Median', 'Mode', 'Minimum', 'Maximum', '10th Percentile', '25th Percentile', '50th Percentile', '75th Percentile', '90th Percentile'],
            'Value': [mean, median, mode, min_marks, max_marks, percentiles[0], percentiles[1], percentiles[2], percentiles[3], percentiles[4]]
        }
        summary_df = pd.DataFrame(summary_data)
        
        plt.figure(figsize=(10, 6))
        sns.set(style='whitegrid')
        plt.subplot(111, frame_on=False)
        
        table = plt.table(cellText=summary_df.values,
                        colLabels=summary_df.columns,
                        cellLoc='center',
                        loc='center',
                        bbox=[0, 0.35, 1, 0.6])  # Adjusted bbox for space

        # Style the table
        for i, (stat, value) in enumerate(summary_df.values):
            if stat == 'Minimum':
                table[(i + 1, 0)].set_text_props(color='red')  # Set the Minimum text color to red
                table[(i + 1, 1)].set_text_props(color='red')
            elif stat == 'Maximum':
                table[(i + 1, 0)].set_text_props(color='green')  # Set the Maximum text color to green
                table[(i + 1, 1)].set_text_props(color='green')
            elif 'Percentile' in stat:
                table[(i + 1, 0)].set_text_props(color='blue')  # Set Percentiles text color to grey
                table[(i + 1, 1)].set_text_props(color='blue')
            else:
                table[(i + 1, 0)].set_text_props(weight='bold')  # Set the text for others to bold
                table[(i + 1, 1)].set_text_props(weight='bold')

        # Set font size and scale
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1.2, 1.2)

        plt.axis('off')
        plt.title('Summary Statistics', pad=12, fontsize=16, fontweight='bold')
        
        # Add Disclaimer
        add_disclaimer(plt.gcf())
        
        # Adjust layout
        plt.subplots_adjust(bottom=0.15)  # Adjust bottom margin to fit disclaimer
        pdf.savefig()
        plt.close()

        # Frequency Table of Specific Marks
        highest_marks = df['Marks'].max()
        lowest_marks = df['Marks'].min()
        mark_frequencies = df['Marks'].value_counts()

        # Get the top 5 most frequent marks
        top_5_frequent_marks = mark_frequencies.head(5).index.tolist()

        # Combine the highest, lowest, and most frequent marks into one list
        marks_of_interest = [highest_marks, lowest_marks] + top_5_frequent_marks

        # Get the frequency of these specific marks
        freq_specific_marks = mark_frequencies.reindex(marks_of_interest, fill_value=0).reset_index()
        freq_specific_marks.columns = ['Marks', 'Frequency']

        # Remove NaN values in case there were fewer than 5 most frequent marks
        freq_specific_marks = freq_specific_marks.dropna()
        
        plt.figure(figsize=(10, 6))
        sns.set(style='whitegrid')
        plt.subplot(111, frame_on=False)
        table = plt.table(cellText=freq_specific_marks.values,
                  colLabels=freq_specific_marks.columns,
                  cellLoc='center',
                  loc='center',
                  bbox=[0, 0.35, 1, 0.6])  # Adjusted bbox for space

        # Style the table
        for i, (mark, freq) in enumerate(freq_specific_marks.values):
            if mark == highest_marks:
                table[(i + 1, 0)].set_text_props(color='green')  # Set the highest mark text color to green
                table[(i + 1, 1)].set_text_props(color='green')
            elif mark == lowest_marks:
                table[(i + 1, 0)].set_text_props(color='red')  # Set the lowest mark text color to red
                table[(i + 1, 1)].set_text_props(color='red')
            table[(i + 1, 0)].set_text_props(weight='bold')  # Set the text for others to bold
            table[(i + 1, 1)].set_text_props(weight='bold')

        # Set font size and scale
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1.2, 1.2)

        plt.axis('off')
        plt.title('Frequency of Highest, Lowest, and Top 5 Most Frequent Marks', pad=12, fontsize=16, fontweight='bold')
        
        # Add Disclaimer
        add_disclaimer(plt.gcf())
        
        # Adjust layout
        plt.subplots_adjust(bottom=0.15)  # Adjust bottom margin to fit disclaimer
        pdf.savefig()
        plt.close()

        # Histogram
        plt.figure(figsize=(10, 7))
        sns.set(style='whitegrid')
        plt.hist(df['Marks'], bins=60, edgecolor='dodgerblue', color='lightgreen')
        plt.title('Distribution of Marks', fontsize=16, fontweight='bold')
        plt.xlabel('Marks', fontsize=14)
        plt.ylabel('Frequency', fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Add Disclaimer
        add_disclaimer(plt.gcf())
        
        # Adjust layout
        plt.subplots_adjust(bottom=0.15)  # Adjust bottom margin to fit disclaimer
        pdf.savefig()
        plt.close()
        
        # Density Plot
        plt.figure(figsize=(10, 7))
        sns.set(style='whitegrid')
        sns.kdeplot(df['Marks'], shade=True, color='darkorange', linewidth=2)
        plt.title('Density of Marks', fontsize=16, fontweight='bold')
        plt.xlabel('Marks', fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Add Disclaimer
        add_disclaimer(plt.gcf())
        
        # Adjust layout
        plt.subplots_adjust(bottom=0.15)  # Adjust bottom margin to fit disclaimer
        pdf.savefig()
        plt.close()

        # Outliers Table
        df['Z-Score'] = (df['Marks'] - mean) / df['Marks'].std()
        outliers = df[np.abs(df['Z-Score']) > 2]
        outliers = outliers.sort_values(by='Marks', ascending=False)
        
        plt.figure(figsize=(10, 6))
        sns.set(style='whitegrid')
        plt.subplot(111, frame_on=False)
        table = plt.table(cellText=outliers[['Serial Number', 'Marks']].values[:12],
                        colLabels=['Serial Number', 'Marks'],
                        cellLoc='center',
                        loc='center',
                        bbox=[0, 0.35, 1, 0.6])  # Adjusted bbox for space
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1.2, 1.2)
        plt.axis('off')
        plt.title('Outlier scores with highest variance (upto top 12)', pad=12, fontsize=16, fontweight='bold')
        
        # Add Disclaimer
        add_disclaimer(plt.gcf())
        
        # Adjust layout
        plt.subplots_adjust(bottom=0.15)  # Adjust bottom margin to fit disclaimer
        pdf.savefig()
        plt.close()
        
        # Box Plot with Outliers Highlighted
        plt.figure(figsize=(10, 7))
        sns.set(style='whitegrid')
        sns.boxplot(x=df['Marks'], color='dodgerblue')
        plt.scatter(outliers['Marks'], np.ones(len(outliers)), color='red', label='Outliers', zorder=5)
        plt.title('Marks with Outliers Highlighted', fontsize=16, fontweight='bold')
        plt.xlabel('Marks', fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        
        # Add Disclaimer
        add_disclaimer(plt.gcf())
        
        # Adjust layout
        plt.subplots_adjust(bottom=0.15)  # Adjust bottom margin to fit disclaimer
        pdf.savefig()
        plt.close()

    print(f"PDF saved as {pdf_filename}")


def generate_summary(input_pdf_file, output_pdf_file):
    df, centre_info = extract_data_from_pdf(input_pdf_file)
    return analyze_and_save_to_pdf(df, centre_info, output_pdf_file)

@app.route('/')
def upload():
    return render_template('index.html')

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def process_pdf(input_pdf_path):
    centre_code = input_pdf_path.split('/')[-1].split('.')[0]
    output_name = f"{centre_code}_centre_analysis_neetug24.pdf"
    output_pdf_path = os.path.join(app.config['PROCESSED_FOLDER'], output_name)
    # print(output_pdf_path)
     # Check if the output file already exists
    if os.path.isfile(output_pdf_path):
        # print("Output file already exists. Skipping processing.")
        return output_pdf_path
    
    # Example PDF processing (you can replace this with your actual processing logic)
    with open(input_pdf_path, 'rb') as input_pdf_file:
        reader = PdfReader(input_pdf_file)
        generate_summary(input_pdf_file=input_pdf_path, output_pdf_file=output_pdf_path)

    return output_pdf_path

@app.route('/readpdf')
def readpdf():
    return render_template('readpdf.html')

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
            return send_file(file_path)
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
