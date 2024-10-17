import os
import base64
import requests
from flask import Flask, request, jsonify, render_template, redirect, url_for
from werkzeug.utils import secure_filename
from flask_cors import CORS

app = Flask(__name__, static_folder='static')

# CORS 설정
CORS(app, resources={r"/*": {"origins": "*"}})

UPLOAD_FOLDER = os.path.join(app.root_path, 'static', 'uploads')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Hugging Face API 정보
API_URL = "https://api-inference.huggingface.co/models/patrickjohncyh/fashion-clip"
headers = {"Authorization": "Bearer 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"}

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
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file part"}), 400
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            # 분석할 candidate labels 설정
            candidate_labels = ["shirt", "pants", "skirt", "jacket", "dress"]

            # Fashion-CLIP 호출
            output = query_fashion_clip(file_path, candidate_labels)

            if output:
                # 가장 높은 점수를 가진 clothing label 찾기
                highest_clothing_label = max(output, key=lambda x: x['score'])['label']

                image_url = f'/static/uploads/{filename}'

                # 상의와 하의에 따라 적절한 결과 페이지로 리디렉션
                if highest_clothing_label in ["shirt", "jacket"]:
                    return jsonify({"success": True, "redirect_url": "/top_analyze", 
                                    "image_url": image_url, "result_sentence": highest_clothing_label})
                else:
                    return jsonify({"success": True, "redirect_url": "/bottom_analyze", 
                                    "image_url": image_url, "result_sentence": highest_clothing_label})
            else:
                return jsonify({"error": "Failed to analyze image"}), 500
        else:
            return jsonify({"error": "Invalid file type"}), 400
    except Exception as e:
        app.logger.error(f"Unexpected error occurred: {e}")
        return jsonify({"error": f"Server error: {str(e)}"}), 500

@app.route('/top_analyze')
def result_top():
    image_url = request.args.get('image_url')
    result_sentence = request.args.get('result_sentence')
    return render_template('top_analyze.html', image_url=image_url, result_sentence=result_sentence)

@app.route('/bottom_analyze')
def result_bottom():
    image_url = request.args.get('image_url')
    result_sentence = request.args.get('result_sentence')
    return render_template('bottom_analyze.html', image_url=image_url, result_sentence=result_sentence)




if __name__ == '__main__':
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.run(debug=True)












