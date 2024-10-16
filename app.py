import os
import base64
import requests
from flask import Flask, request, jsonify, url_for, send_from_directory, render_template
from werkzeug.utils import secure_filename
from flask_cors import CORS
import pandas as pd

app = Flask(__name__, static_folder='static')
CORS(app)
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

def get_top_label(output):
    if output and isinstance(output, list) and len(output) > 0:
        top_item = max(output, key=lambda x: x['score'])
        return top_item['label']
    return None

# 엑셀에서 추천된 상의/하의/색상을 가져오는 함수
def get_top_recommendations(clothing_item, df):
    if clothing_item is None:
        return ['추천 결과 없음'] * 3

    filtered_df = df[df['antecedents'].str.contains(clothing_item, case=False, na=False)]
    
    if filtered_df.empty:
        return ['추천 결과 없음'] * 3

    sorted_df = filtered_df.sort_values(by='lift', ascending=False)
    top_3 = sorted_df['consequents'].head(3).tolist()

    return top_3 + ['추천 결과 없음'] * (3 - len(top_3))

# 이미지 URL을 만들기 위한 함수
def get_image_url(image_name):
    return url_for('uploaded_file', filename=image_name, _external=True)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# 기본 경로에 대한 라우트 추가
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/result/top')
def result_top():
    combined_recommendation_1 = request.args.get('combined_recommendation_1')
    combined_recommendation_2 = request.args.get('combined_recommendation_2')
    combined_recommendation_3 = request.args.get('combined_recommendation_3')
    recommended_image_1 = request.args.get('recommended_image_1')
    recommended_image_2 = request.args.get('recommended_image_2')
    recommended_image_3 = request.args.get('recommended_image_3')

    return render_template('top_analyze.html', 
                           combined_recommendation_1=combined_recommendation_1,
                           combined_recommendation_2=combined_recommendation_2,
                           combined_recommendation_3=combined_recommendation_3,
                           recommended_image_1=recommended_image_1,
                           recommended_image_2=recommended_image_2,
                           recommended_image_3=recommended_image_3)

@app.route('/result/bottom')
def result_bottom():
    combined_recommendation_1 = request.args.get('combined_recommendation_1')
    combined_recommendation_2 = request.args.get('combined_recommendation_2')
    combined_recommendation_3 = request.args.get('combined_recommendation_3')
    recommended_image_1 = request.args.get('recommended_image_1')
    recommended_image_2 = request.args.get('recommended_image_2')
    recommended_image_3 = request.args.get('recommended_image_3')

    return render_template('bottom_analyze.html', 
                           combined_recommendation_1=combined_recommendation_1,
                           combined_recommendation_2=combined_recommendation_2,
                           combined_recommendation_3=combined_recommendation_3,
                           recommended_image_1=recommended_image_1,
                           recommended_image_2=recommended_image_2,
                           recommended_image_3=recommended_image_3)

if __name__ == '__main__':
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.run(debug=True)
