import os
import base64
import requests
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
from flask_cors import CORS
import pandas as pd

app = Flask(__name__, static_folder='static')

# CORS 설정: 모든 경로와 모든 출처에 대해 허용
CORS(app, resources={r"/*": {"origins": "*"}})

UPLOAD_FOLDER = os.path.join(app.root_path, 'static', 'uploads')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Hugging Face API 정보
API_URL = "https://api-inference.huggingface.co/models/patrickjohncyh/fashion-clip"
headers = {"Authorization": "Bearer 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"}

# 엑셀 파일 경로 (상의, 하의, 색상 추천 데이터를 위한 경로)
clothing_recommendation_file = os.path.join(app.root_path, 'deploy', 'clothes.xlsx')
color_recommendation_file = os.path.join(app.root_path, 'deploy', 'color.xlsx')
image_recommendation_file = os.path.join(app.root_path, 'deploy', 'image.xlsx')

# 엑셀 파일 불러오기
clothing_recommendation_df = pd.read_excel(clothing_recommendation_file, sheet_name='Sheet1')
color_recommendation_df = pd.read_excel(color_recommendation_file, sheet_name='Sheet1')
image_df = pd.read_excel(image_recommendation_file, sheet_name='Final')

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def query_fashion_clip(image_path, candidate_labels):
    try:
        with open(image_path, "rb") as f:
            img = f.read()
        payload = {
            "parameters": {"candidate_labels": candidate_labels},
            "inputs": base64.b64encode(img).decode("utf-8")
        }
        response = requests.post(API_URL, headers=headers, json=payload)
        if response.status_code != 200:
            app.logger.error(f"API 요청 실패: {response.status_code}")
            return None
        return response.json()
    except Exception as e:
        app.logger.error(f"API 호출 중 오류 발생: {e}")
        return None

@app.route('/')
def home():
    return "API Server is running"

@app.route('/analyze', methods=['POST', 'OPTIONS'])
def analyze():
    if request.method == 'OPTIONS':
        return jsonify({"status": "OK"}), 200

    try:
        if 'file' not in request.files:
            app.logger.error('File part missing in the request')
            return jsonify({"error": "No file part"}), 400
        file = request.files['file']
        if file.filename == '':
            app.logger.error('No selected file')
            return jsonify({"error": "No selected file"}), 400
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            # Example candidate labels for clothing
            candidate_labels = ["shirt", "pants", "skirt", "jacket", "dress"]
            output = query_fashion_clip(file_path, candidate_labels)

            if output:
                app.logger.info(f'Analysis result: {output}')
                # 분석 결과에 따라 적절한 경로로 리디렉션
                return jsonify({"success": True, "redirect_url": "/top_analyze.html"})
            else:
                app.logger.error('Failed to analyze image, no output from API.')
                return jsonify({"error": "Failed to analyze image"}), 500
        else:
            app.logger.error('Invalid file type')
            return jsonify({"error": "Invalid file type"}), 400
    except Exception as e:
        app.logger.error(f'Unexpected error occurred: {e}')
        return jsonify({"error": f"Server error: {str(e)}"}), 500

# 결과 페이지 처리
@app.route('/top_analyze.html')
def result_top():
    return render_template('top_analyze.html')

@app.route('/bottom_analyze.html')
def result_bottom():
    return render_template('bottom_analyze.html')

if __name__ == '__main__':
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.run(debug=True)


