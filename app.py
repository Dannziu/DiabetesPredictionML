from flask import Flask, render_template, request, jsonify, redirect, url_for, session
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
import joblib
import numpy as np
import json
from datetime import datetime
import os

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-change-in-production'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///diabetes_clinic.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

# Load the trained model
try:
    model = joblib.load('diabetes_pipeline.joblib')
except Exception as e:
    print(f"Warning: Could not load model - {e}")
    model = None

# ============= DATABASE MODELS =============
class Doctor(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    full_name = db.Column(db.String(120), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    medical_registration_number = db.Column(db.String(50), unique=True, nullable=False)
    license_number = db.Column(db.String(50), unique=True, nullable=False)
    password = db.Column(db.String(255), nullable=False)
    specialization = db.Column(db.String(120), nullable=True)
    clinic_name = db.Column(db.String(120), nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    patients = db.relationship('PatientRecord', backref='doctor', lazy=True, cascade='all, delete-orphan')

    def set_password(self, password):
        self.password = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password, password)

class PatientRecord(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    doctor_id = db.Column(db.Integer, db.ForeignKey('doctor.id'), nullable=False)
    patient_name = db.Column(db.String(120), nullable=False)
    age = db.Column(db.Integer, nullable=False)
    pregnancies = db.Column(db.Integer, nullable=True)
    glucose = db.Column(db.Float, nullable=False)
    blood_pressure = db.Column(db.Float, nullable=False)
    skin_thickness = db.Column(db.Float, nullable=False)
    insulin = db.Column(db.Float, nullable=False)
    bmi = db.Column(db.Float, nullable=False)
    diabetes_pedigree = db.Column(db.Float, nullable=False)
    prediction = db.Column(db.Float, nullable=False)
    risk_level = db.Column(db.String(20), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def to_dict(self):
        return {
            'id': self.id,
            'patient_name': self.patient_name,
            'age': self.age,
            'glucose': self.glucose,
            'blood_pressure': self.blood_pressure,
            'bmi': self.bmi,
            'prediction': round(self.prediction, 2),
            'risk_level': self.risk_level,
            'created_at': self.created_at.strftime('%Y-%m-%d %H:%M')
        }

class Admin(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    medical_registration_number = db.Column(db.String(50), unique=True, nullable=False)
    password = db.Column(db.String(255), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def set_password(self, password):
        self.password = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password, password)

# ============= ROUTES =============
@app.route('/')
def index():
    # Redirect logged-in users to dashboard
    if 'doctor_id' in session:
        return redirect(url_for('prediction_dashboard'))
    return render_template('index.html')

@app.route('/doctor/register', methods=['GET', 'POST'])
def doctor_register():
    # Require admin login to register doctors
    if 'admin_id' not in session:
        return redirect(url_for('admin_login'))
    
    if request.method == 'POST':
        data = request.get_json()
        
        if Doctor.query.filter_by(email=data['email']).first():
            return jsonify({'success': False, 'message': 'Email already registered'}), 400
        
        if Doctor.query.filter_by(medical_registration_number=data['medical_registration_number']).first():
            return jsonify({'success': False, 'message': 'Medical registration number already registered'}), 400
        
        if Doctor.query.filter_by(license_number=data['license_number']).first():
            return jsonify({'success': False, 'message': 'License number already registered'}), 400
        
        doctor = Doctor(
            full_name=data['full_name'],
            email=data['email'],
            medical_registration_number=data['medical_registration_number'],
            license_number=data['license_number'],
            specialization=data.get('specialization', ''),
            clinic_name=data.get('clinic_name', '')
        )
        doctor.set_password(data['password'])
        
        db.session.add(doctor)
        db.session.commit()
        
        return jsonify({'success': True, 'message': 'Doctor registration successful!'}), 201
    
    return render_template('doctor_register.html')

@app.route('/doctor/login', methods=['GET', 'POST'])
def doctor_login():
    if request.method == 'POST':
        data = request.get_json()
        doctor = Doctor.query.filter_by(medical_registration_number=data['medical_registration_number']).first()
        
        if doctor and doctor.check_password(data['password']):
            session['doctor_id'] = doctor.id
            session['doctor_name'] = doctor.full_name
            return jsonify({'success': True, 'redirect': url_for('prediction_dashboard')}), 200
        
        return jsonify({'success': False, 'message': 'Invalid medical registration number or password'}), 401
    
    return render_template('doctor_login.html')

@app.route('/prediction', methods=['GET', 'POST'])
def prediction_dashboard():
    if 'doctor_id' not in session:
        return redirect(url_for('doctor_login'))
    
    if request.method == 'POST':
        if not model:
            return jsonify({'success': False, 'message': 'Model not loaded'}), 500
        
        data = request.get_json()
        
        try:
            # Prepare features for prediction
            features = np.array([[
                data['pregnancies'],
                data['glucose'],
                data['blood_pressure'],
                data['skin_thickness'],
                data['insulin'],
                data['bmi'],
                data['diabetes_pedigree'],
                data['age']
            ]])
            
            # Make prediction
            prediction = model.predict_proba(features)[0][1]
            risk_level = 'High Risk' if prediction > 0.6 else 'Moderate Risk' if prediction > 0.4 else 'Low Risk'
            
            # Save patient record
            patient = PatientRecord(
                doctor_id=session['doctor_id'],
                patient_name=data['patient_name'],
                age=data['age'],
                pregnancies=data['pregnancies'],
                glucose=data['glucose'],
                blood_pressure=data['blood_pressure'],
                skin_thickness=data['skin_thickness'],
                insulin=data['insulin'],
                bmi=data['bmi'],
                diabetes_pedigree=data['diabetes_pedigree'],
                prediction=prediction,
                risk_level=risk_level
            )
            
            db.session.add(patient)
            db.session.commit()
            
            return jsonify({
                'success': True,
                'prediction': round(prediction, 2),
                'risk_level': risk_level,
                'message': f'Prediction: {risk_level}'
            }), 200
        
        except Exception as e:
            return jsonify({'success': False, 'message': f'Error: {str(e)}'}), 500
    
    return render_template('prediction.html', doctor_name=session.get('doctor_name'))

@app.route('/doctor/patients')
def doctor_patients():
    if 'doctor_id' not in session:
        return redirect(url_for('doctor_login'))
    
    doctor_id = session['doctor_id']
    patients = PatientRecord.query.filter_by(doctor_id=doctor_id).all()
    
    stats = {
        'total_patients': len(patients),
        'high_risk': sum(1 for p in patients if p.risk_level == 'High Risk'),
        'moderate_risk': sum(1 for p in patients if p.risk_level == 'Moderate Risk'),
        'low_risk': sum(1 for p in patients if p.risk_level == 'Low Risk')
    }
    
    return render_template('doctor_patients.html', patients=patients, stats=stats)

@app.route('/api/patients')
def get_patients():
    if 'doctor_id' not in session:
        return jsonify({'success': False}), 401
    
    doctor_id = session['doctor_id']
    patients = PatientRecord.query.filter_by(doctor_id=doctor_id).all()
    
    return jsonify({
        'success': True,
        'patients': [p.to_dict() for p in patients]
    }), 200

@app.route('/admin/global-history')
def global_history():
    all_patients = PatientRecord.query.all()
    total_records = len(all_patients)
    avg_prediction = np.mean([p.prediction for p in all_patients]) if all_patients else 0
    
    stats = {
        'total_records': total_records,
        'high_risk_count': sum(1 for p in all_patients if p.risk_level == 'High Risk'),
        'avg_prediction': round(avg_prediction, 2)
    }
    
    return render_template('global_history.html', patients=all_patients, stats=stats)

@app.route('/admin/register', methods=['GET', 'POST'])
def admin_register():
    # Check if admin already exists
    existing_admin = Admin.query.first()
    if existing_admin:
        # If admin exists and they access GET, redirect to login
        if request.method == 'GET':
            return redirect(url_for('admin_login'))
        # If admin exists and they POST, reject
        else:
            return jsonify({'success': False, 'message': 'Admin account already exists. Please login instead.'}), 400
    
    if request.method == 'POST':
        try:
            data = request.get_json()
            
            # Validate input
            if not data or 'medical_registration_number' not in data or 'password' not in data:
                return jsonify({'success': False, 'message': 'Medical registration number and password are required'}), 400
            
            medical_reg_num = data.get('medical_registration_number', '').strip()
            password = data.get('password', '')
            
            if not medical_reg_num:
                return jsonify({'success': False, 'message': 'Medical registration number cannot be empty'}), 400
            
            if len(password) < 8:
                return jsonify({'success': False, 'message': 'Password must be at least 8 characters'}), 400
            
            # Double-check no admin exists (race condition protection)
            if Admin.query.first():
                return jsonify({'success': False, 'message': 'Admin already registered'}), 400
            
            # Create admin
            admin = Admin(medical_registration_number=medical_reg_num)
            admin.set_password(password)
            
            db.session.add(admin)
            db.session.commit()
            
            return jsonify({'success': True, 'message': 'Admin registration successful! Please login.'}), 201
        
        except Exception as e:
            db.session.rollback()
            print(f"Admin registration error: {str(e)}")
            return jsonify({'success': False, 'message': 'An error occurred during registration. Please try again.'}), 500
    
    return render_template('admin_register.html')

@app.route('/admin/login', methods=['GET', 'POST'])
def admin_login():
    if request.method == 'POST':
        data = request.get_json()
        admin = Admin.query.filter_by(medical_registration_number=data['medical_registration_number']).first()
        
        if admin and admin.check_password(data['password']):
            session['admin_id'] = admin.id
            return jsonify({'success': True, 'redirect': url_for('admin_panel')}), 200
        
        return jsonify({'success': False, 'message': 'Invalid medical registration number or password'}), 401
    
    return render_template('admin_login.html')

@app.route('/admin/panel')
def admin_panel():
    if 'admin_id' not in session:
        return redirect(url_for('admin_login'))
    
    doctors = Doctor.query.all()
    doctors_data = []
    
    for doctor in doctors:
        patient_count = PatientRecord.query.filter_by(doctor_id=doctor.id).count()
        doctors_data.append({
            'id': doctor.id,
            'full_name': doctor.full_name,
            'email': doctor.email,
            'license_number': doctor.license_number,
            'specialization': doctor.specialization,
            'clinic_name': doctor.clinic_name,
            'patient_count': patient_count,
            'created_at': doctor.created_at.strftime('%Y-%m-%d %H:%M')
        })
    
    return render_template('admin_panel.html', doctors=doctors_data)

@app.route('/api/admin/delete-doctor/<int:doctor_id>', methods=['DELETE'])
def delete_doctor(doctor_id):
    if 'admin_id' not in session:
        return jsonify({'success': False, 'message': 'Unauthorized'}), 401
    
    doctor = Doctor.query.get(doctor_id)
    
    if not doctor:
        return jsonify({'success': False, 'message': 'Doctor not found'}), 404
    
    try:
        # Delete all associated patient records
        PatientRecord.query.filter_by(doctor_id=doctor_id).delete()
        # Delete the doctor
        db.session.delete(doctor)
        db.session.commit()
        
        return jsonify({'success': True, 'message': 'Doctor deleted successfully'}), 200
    
    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'message': f'Error: {str(e)}'}), 500

@app.route('/admin/logout')
def admin_logout():
    session.clear()
    return redirect(url_for('index'))

@app.route('/doctor/logout')
def logout():
    session.clear()
    return redirect(url_for('index'))

# ============= ERROR HANDLERS =============
@app.errorhandler(404)
def not_found(error):
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_error(error):
    db.session.rollback()
    return render_template('500.html'), 500

# ============= DATABASE INITIALIZATION =============
def init_db():
    with app.app_context():
        db.create_all()
        print("Database initialized successfully!")

if __name__ == '__main__':
    init_db()
    app.run(debug=True, host='0.0.0.0', port=5000)
