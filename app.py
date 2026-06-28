import os
import pickle
import numpy as np
from flask import Flask, request, jsonify, render_template_string

# Initialize Flask globally
app = Flask(__name__)

# Load the model file
MODEL_PATH = "model.pkl"
try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
except Exception as e:
    print(f"Error loading {MODEL_PATH}: {str(e)}")
    model = None

# Order of features strictly expected by your XGBoost model (22 features total)
FEATURE_NAMES = [
    "person_age", "person_income", "person_emp_exp", "loan_amnt", 
    "loan_int_rate", "loan_percent_income", "cb_person_cred_hist_length", "credit_score",
    "person_gender_male", "person_education_Bachelor", "person_education_Doctorate",
    "person_education_High School", "person_education_Master", "person_home_ownership_OTHER",
    "person_home_ownership_OWN", "person_home_ownership_RENT", "loan_intent_EDUCATION",
    "loan_intent_HOMEIMPROVEMENT", "loan_intent_MEDICAL", "loan_intent_PERSONAL",
    "loan_intent_VENTURE", "previous_loan_defaults_on_file_Yes"
]

def process_features(data: dict) -> list:
    """Structures raw input data directly into a 22-element list without using Pandas."""
    loan_percent_inc = float(data["loan_amnt"]) / max(float(data["person_income"]), 1.0)
    
    encoded = {
        "person_age": float(data["person_age"]),
        "person_income": float(data["person_income"]),
        "person_emp_exp": int(data["person_emp_exp"]),
        "loan_amnt": float(data["loan_amnt"]),
        "loan_int_rate": float(data["loan_int_rate"]),
        "loan_percent_income": loan_percent_inc,
        "cb_person_cred_hist_length": float(data["cb_person_cred_hist_length"]),
        "credit_score": int(data["credit_score"]),
        "person_gender_male": 1 if data["person_gender"].lower() == "male" else 0,
        "person_education_Bachelor": 1 if "bachelor" in data["person_education"].lower() else 0,
        "person_education_Doctorate": 1 if "doctorate" in data["person_education"].lower() else 0,
        "person_education_High School": 1 if "high school" in data["person_education"].lower() else 0,
        "person_education_Master": 1 if "master" in data["person_education"].lower() else 0,
        "person_home_ownership_OTHER": 1 if data["person_home_ownership"].lower() == "other" else 0,
        "person_home_ownership_OWN": 1 if data["person_home_ownership"].lower() == "own" else 0,
        "person_home_ownership_RENT": 1 if data["person_home_ownership"].lower() == "rent" else 0,
        "loan_intent_EDUCATION": 1 if data["loan_intent"].lower() == "education" else 0,
        "loan_intent_HOMEIMPROVEMENT": 1 if data["loan_intent"].lower() == "homeimprovement" else 0,
        "loan_intent_MEDICAL": 1 if data["loan_intent"].lower() == "medical" else 0,
        "loan_intent_PERSONAL": 1 if data["loan_intent"].lower() == "personal" else 0,
        "loan_intent_VENTURE": 1 if data["loan_intent"].lower() == "venture" else 0,
        "previous_loan_defaults_on_file_Yes": 1 if data["previous_loan_defaults"].lower() == "yes" else 0
    }
    return [encoded[col] for col in FEATURE_NAMES]

# HTML Frontend UI served via base render route
HTML_FRONTEND = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Loan Approval Predictor</title>
    <script src="https://cdn.tailwindcss.com"></script>
</head>
<body class="bg-slate-50 text-slate-800 font-sans min-h-screen flex items-center justify-center py-10 px-4">
    <div class="bg-white p-8 rounded-2xl shadow-xl w-full max-w-2xl border border-slate-100">
        <h2 class="text-3xl font-extrabold text-indigo-700 text-center mb-2">💰 Loan Approval Portal (Flask)</h2>
        <p class="text-center text-slate-500 mb-8">Enter applicant criteria to evaluate risk and eligibility</p>
        
        <form id="predictionForm" class="space-y-6">
            <div class="grid grid-cols-1 sm:grid-cols-2 gap-4">
                <div>
                    <label class="block text-sm font-semibold mb-1">Age</label>
                    <input type="number" name="person_age" value="28" required class="w-full p-2.5 border rounded-lg focus:ring-2 focus:ring-indigo-500 outline-none">
                </div>
                <div>
                    <label class="block text-sm font-semibold mb-1">Annual Income ($)</label>
                    <input type="number" name="person_income" value="65000" required class="w-full p-2.5 border rounded-lg focus:ring-2 focus:ring-indigo-500 outline-none">
                </div>
                <div>
                    <label class="block text-sm font-semibold mb-1">Employment Experience (Years)</label>
                    <input type="number" name="person_emp_exp" value="5" required class="w-full p-2.5 border rounded-lg focus:ring-2 focus:ring-indigo-500 outline-none">
                </div>
                <div>
                    <label class="block text-sm font-semibold mb-1">Loan Requested Amount ($)</label>
                    <input type="number" name="loan_amnt" value="12000" required class="w-full p-2.5 border rounded-lg focus:ring-2 focus:ring-indigo-500 outline-none">
                </div>
                <div>
                    <label class="block text-sm font-semibold mb-1">Interest Rate (%)</label>
                    <input type="number" step="0.01" name="loan_int_rate" value="10.5" required class="w-full p-2.5 border rounded-lg focus:ring-2 focus:ring-indigo-500 outline-none">
                </div>
                <div>
                    <label class="block text-sm font-semibold mb-1">Credit History Length (Years)</label>
                    <input type="number" name="cb_person_cred_hist_length" value="4" required class="w-full p-2.5 border rounded-lg focus:ring-2 focus:ring-indigo-500 outline-none">
                </div>
                <div>
                    <label class="block text-sm font-semibold mb-1">Credit Score (300-850)</label>
                    <input type="number" name="credit_score" value="710" required class="w-full p-2.5 border rounded-lg focus:ring-2 focus:ring-indigo-500 outline-none">
                </div>
                <div>
                    <label class="block text-sm font-semibold mb-1">Gender</label>
                    <select name="person_gender" class="w-full p-2.5 border rounded-lg outline-none">
                        <option value="male">Male</option>
                        <option value="female">Female</option>
                    </select>
                </div>
                <div>
                    <label class="block text-sm font-semibold mb-1">Education Level</label>
                    <select name="person_education" class="w-full p-2.5 border rounded-lg outline-none">
                        <option value="Bachelor">Bachelor</option>
                        <option value="Master">Master</option>
                        <option value="High School">High School</option>
                        <option value="Doctorate">Doctorate</option>
                    </select>
                </div>
                <div>
                    <label class="block text-sm font-semibold mb-1">Home Ownership</label>
                    <select name="person_home_ownership" class="w-full p-2.5 border rounded-lg outline-none">
                        <option value="RENT">RENT</option>
                        <option value="OWN">OWN</option>
                        <option value="MORTGAGE">MORTGAGE</option>
                        <option value="OTHER">OTHER</option>
                    </select>
                </div>
                <div>
                    <label class="block text-sm font-semibold mb-1">Loan Intent</label>
                    <select name="loan_intent" class="w-full p-2.5 border rounded-lg outline-none">
                        <option value="EDUCATION">Education</option>
                        <option value="PERSONAL">Personal</option>
                        <option value="MEDICAL">Medical</option>
                        <option value="VENTURE">Venture</option>
                        <option value="HOMEIMPROVEMENT">Home Improvement</option>
                    </select>
                </div>
                <div>
                    <label class="block text-sm font-semibold mb-1">Historical Default on File?</label>
                    <select name="previous_loan_defaults" class="w-full p-2.5 border rounded-lg outline-none">
                        <option value="No">No</option>
                        <option value="Yes">Yes</option>
                    </select>
                </div>
            </div>
            
            <button type="submit" class="w-full bg-indigo-600 hover:bg-indigo-700 text-white font-bold p-3.5 rounded-lg transition shadow-md">
                Evaluate Loan Eligibility
            </button>
        </form>

        <div id="resultBox" class="mt-8 hidden p-5 rounded-xl border text-center">
            <h3 id="resultStatus" class="text-xl font-bold mb-2"></h3>
            <p id="resultDetails" class="text-sm text-slate-600"></p>
        </div>
    </div>

    <script>
        document.getElementById('predictionForm').addEventListener('submit', async (e) => {
            e.preventDefault();
            const formData = new FormData(e.target);
            const data = Object.fromEntries(formData.entries());
            
            const resultBox = document.getElementById('resultBox');
            const resultStatus = document.getElementById('resultStatus');
            const resultDetails = document.getElementById('resultDetails');
            
            try {
                const response = await fetch('/api/predict', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify(data)
                });
                
                const res = await response.json();
                resultBox.classList.remove('hidden', 'bg-green-50', 'border-green-200', 'bg-red-50', 'border-red-200');
                
                if (res.approved) {
                    resultBox.classList.add('bg-green-50', 'border-green-200', 'text-green-800');
                    resultStatus.innerText = '✅ Approved';
                    resultDetails.innerText = `Applicant is classified as Low-Risk. Risk probability index: ${(res.risk_probability * 100).toFixed(2)}%`;
                } else {
                    resultBox.classList.add('bg-red-50', 'border-red-200', 'text-red-800');
                    resultStatus.innerText = '❌ Denied';
                    resultDetails.innerText = `Applicant failed risk parameters. Default risk metric: ${(res.risk_probability * 100).toFixed(2)}%`;
                }
            } catch (err) {
                alert('Connection failure with machine learning instance.');
            }
        });
    </script>
</body>
</html>
"""

@app.route("/", methods=["GET"])
def index():
    return render_template_string(HTML_FRONTEND)

@app.route("/api/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model payload configuration invalid/missing."}), 500
    
    try:
        raw_data = request.get_json()
        
        # Structure features as a 2D NumPy array directly
        features_list = process_features(raw_data)
        input_features = np.array([features_list])
        
        # Execute prediction trees
        prediction = model.predict(input_features)
        probability = model.predict_proba(input_features)[0][1]
        
        approved_status = bool(prediction[0] == 0)
        
        return jsonify({
            "approved": approved_status,
            "risk_probability": float(probability)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True)
