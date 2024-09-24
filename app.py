import os
import base64
import requests
from flask import Flask, request, render_template, jsonify, url_for, send_from_directory
from werkzeug.utils import secure_filename
from flask_cors import CORS
import pandas as pd
import json
import io
from PIL import Image
from urllib.parse import unquote

app = Flask(__name__, static_folder='static')
CORS(app)
UPLOAD_FOLDER = os.path.join(app.root_path, 'static', 'uploads')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Hugging Face API 정보
API_URL = "https://api-inference.huggingface.co/models/patrickjohncyh/fashion-clip"
headers = {"Authorization": "Bearer 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"}

# FLUX.1-dev API 정보
API_URL_FLUX = "https://api-inference.huggingface.co/models/black-forest-labs/FLUX.1-dev"
headers_flux = {"Authorization": "Bearer hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"}

# 파일 경로 수정
clothing_recommendation_file = os.path.join(app.root_path, 'deploy', 'clothes.xlsx')
color_recommendation_file = os.path.join(app.root_path, 'deploy', 'color.xlsx')

# 엑셀 파일을 불러오기
clothing_recommendation_df = pd.read_excel(clothing_recommendation_file, sheet_name='Sheet1')
color_recommendation_df = pd.read_excel(color_recommendation_file, sheet_name='Sheet1')

# 영어-한국어 매핑 테이블
translation_map = {
    "knitwear": "니트", "shirt": "셔츠", "polo": "카라티", "suit jacket": "블레이저", "T-shirt": "티셔츠",
    "jacket": "자켓", "coat": "코트", "hoodie": "후디", "sweat shirt": "맨투맨",
    "cotton pants": "면바지", "sweat pants": "스웻팬츠", "denim pants": "청바지", "cargo pants": "카고 팬츠",
    "shorts": "반바지", "dress pants": "슬랙스",
    "blue clothes": "파란색", "black clothes": "검은색", "red clothes": "빨간색", "white clothes": "흰색",
    "grey clothes": "회색", "beige clothes": "베이지색", "green clothes": "초록색", "navy clothes": "네이비색"
}

# 매핑 테이블에 라벨을 등록하는 API
@app.route('/add_translation', methods=['POST'])
def add_translation():
    english_word = request.form.get('english_word')
    korean_word = request.form.get('korean_word')
    
    if not english_word or not korean_word:
        return jsonify({"success": False, "error": "영어 또는 한국어 단어가 제공되지 않았습니다."})

    if english_word in translation_map:
        translation_map[english_word] = korean_word
        save_translation_map()
        return jsonify({"success": True, "message": f"'{english_word}'가 '{korean_word}'로 성공적으로 등록되었습니다."})
    
    return jsonify({"success": False, "error": f"'{english_word}'는 등록할 수 없는 라벨입니다."})

# 매핑 테이블을 JSON 파일로 저장
def save_translation_map(filename="translation_map.json"):
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(translation_map, f, ensure_ascii=False, indent=4)

# JSON 파일로부터 매핑 테이블 불러오기
def load_translation_map(filename="translation_map.json"):
    global translation_map
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            translation_map = json.load(f)
    except FileNotFoundError:
        save_translation_map()

load_translation_map()

def map_to_korean(english_text):
    return translation_map.get(english_text, english_text)

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
            return None
        return response.json()
    except Exception as e:
        app.logger.error(f"API 호출 중 오류 발생: {e}")
        return None

def get_top_label(output):
    if output and isinstance(output, list):
        top_item = max(output, key=lambda x: x['score'])
        return top_item['label']
    return None

def get_top_recommendations(clothing_item, df):
    # 필터링된 데이터프레임에서 추천 의류를 가져오는 로직
    filtered_df = df[df['antecedents'].str.contains(clothing_item, case=False, na=False)]
    if filtered_df.empty:
        app.logger.error(f"추천 결과가 없습니다: {clothing_item}")
        return []
    sorted_df = filtered_df.sort_values(by='lift', ascending=False)
    top_3 = sorted_df['consequents'].head(3).tolist()
    app.logger.info(f"추천된 의류 종류: {top_3}")
    return top_3

def get_top_color_recommendations(clothing_item, df):
    # 필터링된 데이터프레임에서 추천 색상을 가져오는 로직
    filtered_df = df[df['antecedents'].str.contains(clothing_item, case=False, na=False)]
    if filtered_df.empty:
        app.logger.error(f"추천 색상 결과가 없습니다: {clothing_item}")
        return []
    sorted_df = filtered_df.sort_values(by='lift', ascending=False)
    top_3 = sorted_df['consequents'].head(3).tolist()
    app.logger.info(f"추천된 색상: {top_3}")
    return top_3


# FLUX.1-dev API로 요청
def query_flux(payload):
    try:
        response = requests.post(API_URL_FLUX, headers=headers_flux, json=payload)
        if response.status_code == 200:
            return response.content
        else:
            app.logger.error(f"FLUX.1-dev API 호출 중 오류 발생: {response.status_code}, 응답: {response.text}")
            return None
    except Exception as e:
        app.logger.error(f"FLUX.1-dev API 호출 중 오류 발생: {e}")
        return None

def generate_outfit_image(prompt):
    image_bytes = query_flux({"inputs": prompt})
    if image_bytes:
        return Image.open(io.BytesIO(image_bytes))
    else:
        return None

def create_image_prompt(clothing_color, clothing_item, recommended_color, recommended_clothing):
    # 색상 라벨에서 "clothes"라는 단어를 제거하여 프롬프트를 더 정확하게 만듦
    clothing_color_cleaned = clothing_color.replace(' clothes', '')  # 'blue clothes' -> 'blue'
    recommended_color_cleaned = recommended_color.replace(' clothes', '')  # 'navy clothes' -> 'navy'
    
    # 프롬프트 생성
    prompt = f"A man who is wearing {clothing_color_cleaned} {clothing_item} and {recommended_color_cleaned} {recommended_clothing}"
    return prompt

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        if 'file' not in request.files or 'clothing-type' not in request.form:
            return jsonify({'success': False, 'error': '파일 또는 의류 종류가 선택되지 않았습니다.'}), 400

        file = request.files['file']
        clothing_type = request.form['clothing-type']

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # 의류 종류 분석
            if clothing_type == "상의":
                candidate_labels = ["knitwear", "shirt", "polo", "suit jacket", "T-shirt", "jacket", "coat", "hoodie", "sweat shirt"]
            else:
                candidate_labels = ["cotton pants", "sweat pants", "denim pants", "cargo pants", "shorts", "dress pants"]

            output = query_fashion_clip(filepath, candidate_labels)
            if output is None:
                return jsonify({'success': False, 'error': 'API 요청에 실패했습니다.'})

            # 영어 라벨 저장
            clothing_label_en = get_top_label(output)

            # 색상 분석
            color_labels = ["blue clothes", "black clothes", "red clothes", "white clothes", "grey clothes", "beige clothes", "green clothes", "navy clothes"]
            color_output = query_fashion_clip(filepath, color_labels)
            if color_output is None:
                return jsonify({'success': False, 'error': '색상 분석 요청에 실패했습니다.'})

            # 영어 색상 저장
            color_label_en = get_top_label(color_output)

            # 의류 및 색상 추천 결과 가져오기 (영어로)
            clothing_recommendations_en = get_top_recommendations(clothing_label_en, clothing_recommendation_df)
            color_recommendations_en = get_top_color_recommendations(color_label_en, color_recommendation_df)

            # 추천 결과가 비어있는지 확인
            if not clothing_recommendations_en or not color_recommendations_en:
                app.logger.error(f"추천 결과가 비어 있습니다: clothing_recommendations_en={clothing_recommendations_en}, color_recommendations_en={color_recommendations_en}")
                return jsonify({'success': False, 'error': '추천 결과가 없습니다.'}), 400

            # 영어 값을 한국어로 매핑
            clothing_label_kr = map_to_korean(clothing_label_en)
            color_label_kr = map_to_korean(color_label_en)
            clothing_recommendations_kr = [map_to_korean(item) for item in clothing_recommendations_en]
            color_recommendations_kr = [map_to_korean(item) for item in color_recommendations_en]

            # URL에 한국어와 영어 라벨 모두 전달
            image_url = f"/uploads/{filename}"

            if clothing_type == '상의':
                return jsonify({
                    'success': True,
                    'label_en': clothing_label_en,
                    'color_en': color_label_en,
                    'clothing_recommendations_en': clothing_recommendations_en,
                    'color_recommendations_en': color_recommendations_en,
                    'label_kr': clothing_label_kr,
                    'color_kr': color_label_kr,
                    'clothing_recommendations_kr': clothing_recommendations_kr,
                    'color_recommendations_kr': color_recommendations_kr,
                    'image_url': image_url,
                    'redirect_url': url_for('result_top', label_en=clothing_label_en, color_en=color_label_en, label_kr=clothing_label_kr, color_kr=color_label_kr, image_url=image_url, clothing_recommendations=','.join(clothing_recommendations_en), color_recommendations=','.join(color_recommendations_en), _external=True)
                })

            else:
                return jsonify({
                    'success': True,
                    'label_en': clothing_label_en,
                    'color_en': color_label_en,
                    'clothing_recommendations_en': clothing_recommendations_en,
                    'color_recommendations_en': color_recommendations_en,
                    'label_kr': clothing_label_kr,
                    'color_kr': color_label_kr,
                    'clothing_recommendations_kr': clothing_recommendations_kr,
                    'color_recommendations_kr': color_recommendations_kr,
                    'image_url': image_url,
                    'redirect_url': url_for('result_bottom', label_en=clothing_label_en, color_en=color_label_en, label_kr=clothing_label_kr, color_kr=color_label_kr, image_url=image_url, clothing_recommendations=','.join(clothing_recommendations_en), color_recommendations=','.join(color_recommendations_en), _external=True)
                })
    except Exception as e:
        app.logger.error(f"서버 내부 오류 발생: {e}")
        return jsonify({'success': False, 'error': '서버 내부 오류입니다.'}), 500

@app.route('/result/top')
def result_top():
    # 전달된 한국어 라벨과 색상 값을 사용해 결과 문장 구성
    label_kr = request.args.get('label_kr', '')  # 한국어 의류 종류 (예: 셔츠)
    color_kr = request.args.get('color_kr', '')  # 한국어 색상 (예: 흰색)
    label_en = request.args.get('label_en', '')  # 영어 의류 종류 (예: shirt)
    color_en = request.args.get('color_en', '')  # 영어 색상 (예: white clothes)
    image_url = request.args.get('image_url')

    # 추천된 하의와 색상 값을 URL에서 디코딩 및 리스트로 변환
    clothing_recommendations_en = unquote(request.args.get('clothing_recommendations', '')).split(',')
    color_recommendations_en = unquote(request.args.get('color_recommendations', '')).split(',')

    # 추천된 값을 한국어로 변환
    clothing_recommendations_kr = [map_to_korean(item) for item in clothing_recommendations_en]
    color_recommendations_kr = [map_to_korean(item) for item in color_recommendations_en]

    # 프롬프트 생성 및 로그 출력
    prompt = create_image_prompt(color_en, label_en, color_recommendations_en[0], clothing_recommendations_en[0])
    app.logger.info(f"생성된 프롬프트 (영어): {prompt}")

    # 이미지 생성
    generated_image = generate_outfit_image(prompt)

    # 생성된 이미지 저장 및 URL 전달
    if generated_image:
        try:
            generated_image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'generated_image.png')
            generated_image.save(generated_image_path)
            generated_image_url = url_for('uploaded_file', filename='generated_image.png')
        except Exception as e:
            app.logger.error(f"이미지 저장 중 오류 발생: {e}")
            generated_image_url = None
    else:
        generated_image_url = None

    # 전달된 값이 비어 있는지 확인
    if not label_kr or not color_kr or not clothing_recommendations_kr or not color_recommendations_kr:
        app.logger.error(f"한국어 라벨, 색상 또는 추천된 값이 전달되지 않았습니다.")
        return jsonify({'error': '추천된 값이 전달되지 않았습니다.'}), 400

    # 분석 결과를 한국어로 구성
    result_sentence = f"분석 결과, 이미지 속 옷은 {color_kr} {label_kr}이며 \n추천 하의 종류는 {', '.join(clothing_recommendations_kr)}이고, 추천 하의 색상은 {', '.join(color_recommendations_kr)}입니다."

    # 결과 페이지로 렌더링
    return render_template('top_analyze.html', result_sentence=result_sentence, image_url=image_url, generated_image_url=generated_image_url)


@app.route('/result/bottom')
def result_bottom():
    # 전달된 한국어 라벨과 색상 값을 사용해 결과 문장 구성
    label_kr = request.args.get('label_kr', '')  # 한국어 의류 종류 (예: 바지)
    color_kr = request.args.get('color_kr', '')  # 한국어 색상 (예: 파란색)
    label_en = request.args.get('label_en', '')  # 영어 의류 종류 (예: denim pants)
    color_en = request.args.get('color_en', '')  # 영어 색상 (예: blue clothes)
    image_url = request.args.get('image_url')

    # 추천된 상의와 색상 값을 URL에서 디코딩 및 리스트로 변환
    clothing_recommendations_en = unquote(request.args.get('clothing_recommendations', '')).split(',')
    color_recommendations_en = unquote(request.args.get('color_recommendations', '')).split(',')

    # 추천된 값을 한국어로 변환
    clothing_recommendations_kr = [map_to_korean(item) for item in clothing_recommendations_en]
    color_recommendations_kr = [map_to_korean(item) for item in color_recommendations_en]

    # 프롬프트 생성 및 로그 출력
    prompt = create_image_prompt(color_en, label_en, color_recommendations_en[0], clothing_recommendations_en[0])
    app.logger.info(f"생성된 프롬프트 (영어): {prompt}")

    # 이미지 생성
    generated_image = generate_outfit_image(prompt)

    # 생성된 이미지 저장 및 URL 전달
    if generated_image:
        try:
            generated_image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'generated_image.png')
            generated_image.save(generated_image_path)
            generated_image_url = url_for('uploaded_file', filename='generated_image.png')
        except Exception as e:
            app.logger.error(f"이미지 저장 중 오류 발생: {e}")
            generated_image_url = None
    else:
        generated_image_url = None

    # 전달된 값이 비어 있는지 확인
    if not label_kr or not color_kr or not clothing_recommendations_kr or not color_recommendations_kr:
        app.logger.error(f"한국어 라벨, 색상 또는 추천된 값이 전달되지 않았습니다.")
        return jsonify({'error': '추천된 값이 전달되지 않았습니다.'}), 400

    # 분석 결과를 한국어로 구성
    result_sentence = f"분석 결과, 이미지 속 옷은 {color_kr} {label_kr}이며 \n추천 상의 종류는 {', '.join(clothing_recommendations_kr)}이고, 추천 상의 색상은 {', '.join(color_recommendations_kr)}입니다."

    # 결과 페이지로 렌더링
    return render_template('bottom_analyze.html', result_sentence=result_sentence, image_url=image_url, generated_image_url=generated_image_url)


if __name__ == '__main__':
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.run(debug=True)
