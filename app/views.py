from app import app
from flask   import render_template, request, redirect,url_for , session , jsonify
from jinja2  import TemplateNotFound
import numpy as np
import csv
import pickle
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
import io
from sklearn.metrics.pairwise import euclidean_distances
import base64
import requests
from io import StringIO

app.secret_key = '12345'
@app.route('/')
def index():
    return render_template('form.html')

@app.route('/formsubmit', methods=['POST'])
def model_data_info():
    # Get the form data
    session.clear()
    gender = request.form.get('gender')
    age = request.form.get('age')
    annual_income = request.form.get('annual_income')
    spending_score = request.form.get('spending_score')
    membership_tier = request.form.get('membership_tier')
    weekly_income = request.form.get('weekly_income')
    region = request.form.get('region')

    # Check if all the fields have been filled
    if not all([gender, age, annual_income, spending_score, membership_tier, weekly_income, region]):
        return "Please fill all the form fields", 400

    # Process or print the data
    print(gender, age, annual_income, spending_score, membership_tier, weekly_income, region)
    new_data = np.array([[float(spending_score), float(membership_tier)]])  # Convert to float if necessary

    # Load the models
    with open('D:\\Kuliah\\DTS Kominfo - Data Science\\CAPSTONEDATASCIENCE-main\\dbscan_model.pkl', 'rb') as f:
        loaded_dbscan = pickle.load(f)
    with open('D:\\Kuliah\\DTS Kominfo - Data Science\\CAPSTONEDATASCIENCE-main\\scaler.pkl', 'rb') as f:
        loaded_scaler = pickle.load(f)

    # Perform prediction or clustering
    new_labels, distances = test_new_data(new_data, loaded_dbscan, loaded_scaler)

    # Prepare the results
    results = []
    dataLabel = []
    dataReward = []
    dataDistance = []
    dataDetails = []
    customer_data = {
        "Gender": gender,
        "Age": age,
        "Annual_Income": annual_income,
        "Membership_Tier": membership_tier,
        "Spending_Score": spending_score,
        "Weekly_Income": weekly_income,
        "Region": region
    }

    # Selalu tambahkan data ke dataDetails
    dataDetails.append(customer_data)

    for i, (label, distance) in enumerate(zip(new_labels, distances)):
        if label == 1 or label == 3:
            reward = "Mendapat Reward"
        else:
            reward = "Tidak Mendapat Reward"
        
        if label != -1:
            text = f"Data point {i} termasuk dalam cluster {label} dengan jarak {distance:.4f} ke titik inti terdekat {reward}"
        else:
            text = f"Data point {i} diklasifikasikan sebagai noise. Cluster terdekat berjarak {distance:.4f} {reward}"

        # Append data (converted to native Python types for JSON compatibility)
        results.append(text)
        dataLabel.append(int(label))  # Convert to int
        dataReward.append(reward)
        dataDistance.append(float(distance))  # Convert to float

    # Simpan hasil ke session
    session['dataLabel'] = dataLabel
    session['dataReward'] = dataReward
    session['dataDistance'] = dataDistance
    session['results'] = results
    session['dataDetails'] = dataDetails
        
    # Redirect ke halaman hasil
    return redirect(url_for('modeldatainfo'))

@app.route('/modeldatainfo')
def modeldatainfo():
    # Ambil data dari session
    data_label = session.get('dataLabel', [])
    data_reward = session.get('dataReward', [])
    data_distance = session.get('dataDistance', [])
    results = session.get('results', [])
    data_details = session.get('dataDetails', [])

    # Log jumlah data yang diambil dari session
    app.logger.info(f"Jumlah item dalam data_details: {len(data_details)}")
    app.logger.info(f"Jumlah item dalam results: {len(results)}")

    # Kembalikan hasil dalam format JSON
    # return jsonify({
    #     'dataLabel': data_label,
    #     'dataReward': data_reward,
    #     'dataDistance': data_distance,
    #     'results': results,
    #     'dataDetails': data_details
    # })
    return render_template('hasil.html' , data_label=data_label, data_reward=data_reward, data_distance=data_distance, results=results, data_details=data_details , len = len(data_label))

@app.route ('/customerdata')
def customer_data_list():
    return 'Customer Data List'

@app.route('/process_csv', methods=['GET'])
def model_data_info_csv():
    try:
        session.clear()
        # Baca file CSV
        url = 'D:\\Kuliah\\DTS Kominfo - Data Science\\CAPSTONEDATASCIENCE-main\\Book1.csv'
        # response = requests.get(url)
        # response.raise_for_status()  # Akan raise exception jika ada HTTP error
        
        # # Log response content untuk debugging
        # app.logger.info(f"CSV content preview: {response.text[:500]}")
        
        df = pd.read_csv(url)  # Gunakan StringIO dari io
        
        # Log DataFrame info untuk debugging
        app.logger.info(f"DataFrame info: {df.info()}")
        app.logger.info(f"DataFrame shape: {df.shape}")
        
        # Proses setiap baris dalam file CSV
        results = []
        dataLabel = []
        dataReward = []
        dataDistance = []
        dataDetails = []

        # Load models
        try:
            with open('D:\\Kuliah\\DTS Kominfo - Data Science\\CAPSTONEDATASCIENCE-main\\dbscan_model.pkl', 'rb') as f:
                loaded_dbscan = pickle.load(f)
            with open('D:\\Kuliah\\DTS Kominfo - Data Science\\CAPSTONEDATASCIENCE-main\\scaler.pkl', 'rb') as f:
                loaded_scaler = pickle.load(f)
        except FileNotFoundError as e:
            app.logger.error(f"Model file not found: {str(e)}")
            return jsonify({"error": "Model file not found"}), 500

        for index, row in df.iterrows():
            # Ekstrak data dari baris
            customer_data = {
                "CustomerID": row['CustomerID'],
                "Gender": row['Gender'],
                "Age": row['Age'],
                "Annual_Income": row['Annual_Income'],
                "Membership_Tier": row['Membership_Tier'],
                "Spending_Score": row['Spending_Score'],
                "Weekly_Income": row['Weekly_Income'],
                "Region": row['Region']
            }

            # Selalu tambahkan data ke dataDetails
            dataDetails.append(customer_data)

            # Periksa apakah semua field memiliki nilai
            if pd.isna(row['Spending_Score']) or pd.isna(row['Membership_Tier']):
                results.append(f"Baris {index + 2}: Data tidak lengkap. Pastikan semua kolom terisi.")
                dataLabel.append(10)
                dataReward.append("Data Tidak Valid")
                dataDistance.append(0.0)
                continue

            # Proses data
            new_data = np.array([[row['Spending_Score'], row['Membership_Tier']]])

            new_labels, distances = test_new_data(new_data, loaded_dbscan, loaded_scaler)

            for i, (label, distance) in enumerate(zip(new_labels, distances)):
                reward = "Mendapat Reward" if label in [1, 3] else "Tidak Mendapat Reward"

                if label != -1:
                    text = f"Baris {index + 2}: Data point {i} termasuk dalam cluster {label} dengan jarak {distance:.4f} ke titik inti terdekat. {reward}"
                else:
                    text = f"Baris {index + 2}: Data point {i} diklasifikasikan sebagai noise. Cluster terdekat berjarak {distance:.4f}. {reward}"

                dataLabel.append(int(label))
                dataReward.append(reward)
                dataDistance.append(float(distance))
                results.append(text)

        # Log jumlah data sebelum disimpan ke session
        app.logger.info(f"Jumlah item dalam dataDetails sebelum disimpan: {len(dataDetails)}")
        app.logger.info(f"Jumlah item dalam results sebelum disimpan: {len(results)}")

        # Simpan hasil ke session
        session['dataLabel'] = dataLabel
        session['dataReward'] = dataReward
        session['dataDistance'] = dataDistance
        session['results'] = results
        session['dataDetails'] = dataDetails
        
        # Log jumlah data setelah disimpan ke session
        app.logger.info(f"Jumlah item dalam session['dataDetails']: {len(session.get('dataDetails', []))}")
        app.logger.info(f"Jumlah item dalam session['results']: {len(session.get('results', []))}")
        
        # Redirect ke halaman hasil
        return redirect(url_for('modeldatainfo'))
    
    except Exception as e:
        app.logger.error(f"Terjadi kesalahan: {str(e)}")
        return jsonify({"error": f"Terjadi kesalahan saat memproses data: {str(e)}"}), 500
    
def test_new_data(new_data, dbscan_model, scaler):
    new_data_scaled = scaler.transform(new_data)
    distances = euclidean_distances(new_data_scaled, dbscan_model.components_)
    min_distances = distances.min(axis=1)
    nearest_core_indices = distances.argmin(axis=1)
    
    new_labels = np.where(min_distances <= dbscan_model.eps, 
                          dbscan_model.labels_[dbscan_model.core_sample_indices_][nearest_core_indices], 
                          -1)
    
    return new_labels, min_distances