import os
from flask import Flask, render_template, request, send_file
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Create uploads folder if it does not exist
        if not os.path.exists(UPLOAD_FOLDER):
            os.makedirs(UPLOAD_FOLDER)

        # Get the uploaded file and save it to the uploads folder
        file = request.files['file']
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], file.filename))

        # Redirect to the histogram page
        return render_template('upload.html', filename=file.filename)
    else:
        return render_template('upload.html')


@app.route('/histogram/<filename>')
def histogram(filename):
    # Read the uploaded file and select the FICO score column
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    data = pd.read_csv(filepath)
    fico_scores = data[' fico']

    # Create a histogram with 20 bins
    plt.hist(fico_scores, bins=20)

    # Add axis labels and a title
    plt.xlabel('FICO score')
    plt.ylabel('Frequency')
    plt.title('Distribution of FICO scores')

    # Save the plot and confusion matrix to a PDF file
    plot_filepath = os.path.join(app.config['UPLOAD_FOLDER'], 'plot.pdf')
    with PdfPages(plot_filepath) as pdf:
        # Plot the histogram and add it to the PDF file
        pdf.savefig()

        # Calculate the confusion matrix for  model_target and  model_output
        binary_target = data[' model_target'].apply(lambda x: 1 if x > 0 else 0)
        binary_output = data[' model_output'].apply(lambda x: 1 if x > 0.5 else 0)
        cm = pd.crosstab(binary_target, binary_output, rownames=['Actual'], colnames=['Predicted'])

        # Plot the confusion matrix and add it to the PDF file
        plt.figure()
        plt.imshow(cm, cmap=plt.cm.Blues, interpolation='nearest')
        plt.colorbar()
        plt.xticks([0, 1], ['No Default', 'Default'])
        plt.yticks([0, 1], ['No Default', 'Default'])
        plt.xlabel('Predicted label')
        plt.ylabel('True label')
        plt.title('Confusion matrix')
        pdf.savefig()

    # Return the plot and confusion matrix as a downloadable file
    return send_file(plot_filepath, as_attachment=True)


if __name__ == '__main__':
    app.run(debug=True)
