from flask import Flask, request, render_template, jsonify
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from io import BytesIO
import base64

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

SAMPLE_DATASET_PATH = '/Users/dswanson/PycharmProjects/pythonProject/hans_app/uploads/PvsV test plot data.xlsx'

def load_and_process_dataset(file_path, nth_pt):
    df = pd.read_excel(file_path)
    data = df.to_dict(orient='records')  # Convert DataFrame to a list of dictionaries
    x_values, y_values = zip(*df.itertuples(index=False, name=None))

    x_values_subset = x_values[nth_pt:]
    y_values_subset = y_values[nth_pt:]
    z = np.polyfit(x_values_subset, y_values_subset, 1)
    p = np.poly1d(z)

    x_extrapolate = np.linspace(min(x_values_subset) * -0.4, max(x_values_subset) * 2.4, 100)
    y_extrapolate = p(x_extrapolate)

    number_moles = (0.04 / 58.12)
    gradient_divided_by_27 = z[0] / (number_moles * 293)

    def round_to_sig_figs(num, sig_figs):
        return round(num, sig_figs - len(str(int(number_moles))) + 1)

    rounded_n = round_to_sig_figs(number_moles, 6)

    plt.figure(figsize=(10, 6))
    plt.scatter(x_values, y_values, label='Data points', color='blue', marker='o')
    plt.plot(x_extrapolate, y_extrapolate, "r--", label=f'Best fit line: y = {z[0]:.2f}x + {z[1]:.2f}\n n = {rounded_n} mol \n T = 293K \n R = gradient / nT = {gradient_divided_by_27:.2f}')
    plt.title('Real Gas Investigation of Pressure vs Inverse Volume of Butane')
    plt.xlabel('1/V (L^-1)')
    plt.ylabel('Pressure (kPa)')
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.minorticks_on()

    img = BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plt.close()

    img_data = base64.b64encode(img.getvalue()).decode('utf-8')
    return img_data, data

@app.route('/', methods=['GET', 'POST'])
def index():
    img_data = None  # Initialize img_data to None
    data = []  # Initialize data to an empty list
    try:
        nth_pt = 4
        file_path = SAMPLE_DATASET_PATH

        if request.method == 'POST':
            uploaded_file = request.files.get('file')
            if uploaded_file and uploaded_file.filename.endswith('.xlsx'):
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], uploaded_file.filename)
                uploaded_file.save(file_path)

            nth_pt = int(request.form.get('nth_pt', 4))

        img_data, data = load_and_process_dataset(file_path, nth_pt)

        if request.headers.get('Accept') == 'application/json':
            return jsonify(img_data=img_data)

        return render_template('index.html', nth_pt=nth_pt, img_data=img_data, data=data)
    except Exception as e:
        return f"An error occurred: {str(e)}"

if __name__ == '__main__':
    img_data, data = load_and_process_dataset(SAMPLE_DATASET_PATH, 4)
    app.run(debug=True)