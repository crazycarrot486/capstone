import os
import base64
import requests
from flask import Flask, request, jsonify, url_for, send_from_directory, render_template
from werkzeug.utils import secure_filename
from flask_cors import CORS
import pandas as pd

app = Flask(__name__, static_folder='static')
CORS(app, resources={r"/*": {"origins": "https://fillout-closet.netlify.app/"}})
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

# 옷과 색상 번역 함수 추가
def translate_to_korean(label):
    translation_dict = {
        "shirt": "셔츠",
        "T-shirt": "티셔츠",
        "polo": "폴로티",
        "suit jacket": "블레이저",
        "knitwear": "니트",
        "jacket": "재킷",
        "coat": "코트",
        "hoodie": "후드",
        "sweat shirt": "맨투맨",
        "cotton pants": "면바지",
        "sweat pants": "스웻 팬츠",
        "denim pants": "청바지",
        "cargo pants": "카고 팬츠",
        "shorts": "쇼츠",
        "dress pants": "슬랙스"
    }
    return translation_dict.get(label, label)

def translate_color_to_korean(color_label):
    color_translation_dict = {
        "blue clothes": "파란색",
        "black clothes": "검은색",
        "red clothes": "빨간색",
        "white clothes": "흰색",
        "grey clothes": "그레이색",
        "beige clothes": "베이지색",
        "green clothes": "초록색",
        "navy clothes": "남색"
    }
    return color_translation_dict.get(color_label, color_label)

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

# 옷 종류와 색상을 기준으로 이미지 찾기
def find_matching_images_for_combinations(combinations, df, clothing_type, analyzed_clothing, analyzed_color):
    matched_images = []
    
    if clothing_type == '상의':
        category_column = 'Max Top Category'
        color_column = '1st Color'
        recommendation_category_column = 'Max Bottom Category'
        recommendation_color_column = '2nd Color'
    
    else:
        category_column = 'Max Bottom Category'
        color_column = '2nd Color'
        recommendation_category_column = 'Max Top Category'
        recommendation_color_column = '1st Color'
    
    for combo in combinations:
        clothing_type, color = combo
        
        # 추천된 조합뿐만 아니라 분석된 의류 종류와 색상도 함께 매칭
        matched_row = df[
            (df[category_column] == analyzed_clothing) & 
            (df[color_column] == analyzed_color) & 
            (df[recommendation_category_column] == clothing_type) & 
            (df[recommendation_color_column] == color)
        ]
        
        if not matched_row.empty:
            image_name = matched_row.iloc[0]['image']
            matched_images.append(get_image_url(image_name))
            app.logger.info(f"추천된 이미지: {image_name}, 분석된 의류 종류: {analyzed_clothing}, 분석된 색상: {analyzed_color}, 추천된 의류: {clothing_type}, 추천된 색상: {color}")
        
        if len(matched_images) == 3:
            break

    while len(matched_images) < 3:
        matched_images.append('/static/default_image.jpg')
    
    return matched_images

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        # 파일과 의류 종류가 전송되었는지 확인
        if 'file' not in request.files or 'clothing-type' not in request.form:
            return jsonify({'success': False, 'error': '파일 또는 의류 종류가 선택되지 않았습니다.'}), 400

        file = request.files['file']
        clothing_type = request.form['clothing-type']

        # 파일 유효성 검사
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # 의류 종류에 따른 분석 후보 설정
            candidate_labels = ["shirt", "T-shirt", "polo", "suit jacket", "knitwear", "jacket", "coat", "hoodie", "sweat shirt"] if clothing_type == "상의" else ["cotton pants", "sweat pants", "denim pants", "cargo pants", "shorts", "dress pants"]

            # 의류 종류 분석
            output = query_fashion_clip(filepath, candidate_labels)
            if output is None:
                return jsonify({'success': False, 'error': 'API 요청에 실패했습니다.'})

            # 분석된 의류 종류 저장 및 한국어 변환
            clothing_label = get_top_label(output)
            clothing_label_korean = translate_to_korean(clothing_label)
            app.logger.info(f"분석된 의류 종류: {clothing_label} -> {clothing_label_korean}")

            # 색상 분석
            color_labels = ["blue clothes", "black clothes", "red clothes", "white clothes", "grey clothes", "beige clothes", "green clothes", "navy clothes"]
            color_output = query_fashion_clip(filepath, color_labels)
            if color_output is None:
                return jsonify({'success': False, 'error': '색상 분석 요청에 실패했습니다.'})

            # 분석된 색상 저장 및 한국어 변환
            color_label = get_top_label(color_output)
            color_label_korean = translate_color_to_korean(color_label)
            app.logger.info(f"분석된 색상: {color_label} -> {color_label_korean}")

            # 의류 및 색상 추천 데이터 생성
            clothing_recommendations = get_top_recommendations(clothing_label, clothing_recommendation_df)
            color_recommendations = get_top_recommendations(color_label, color_recommendation_df)

            # 추천 조합 생성 (9가지)
            combinations = [(clothing_recommendations[i], color_recommendations[j]) for i in range(3) for j in range(3)]
            app.logger.info(f"생성된 조합: {combinations}")

            # 추천 이미지 찾기 (분석된 의류 종류와 색상도 함께 고려)
            recommended_images = find_matching_images_for_combinations(combinations, image_df, clothing_type, clothing_label, color_label)
            app.logger.info(f"추천된 이미지 URL: {recommended_images}")

            # 분석 결과에 대한 JSON 응답 생성
            response_data = {
                'success': True,
                'label': clothing_label,
                'label_korean': clothing_label_korean,
                'color': color_label,
                'color_korean': color_label_korean,
                'combined_recommendation_1': combinations[0],
                'combined_recommendation_2': combinations[1],
                'combined_recommendation_3': combinations[2],
                'image_url': f"/uploads/{filename}",
                'recommended_image_1': recommended_images[0],
                'recommended_image_2': recommended_images[1],
                'recommended_image_3': recommended_images[2],
            }

            # 리다이렉트 URL 설정
            redirect_endpoint = 'result_top' if clothing_type == '상의' else 'result_bottom'
            response_data['redirect_url'] = url_for(redirect_endpoint,
                                                    combined_recommendation_1=combinations[0],
                                                    combined_recommendation_2=combinations[1],
                                                    combined_recommendation_3=combinations[2],
                                                    recommended_image_1=recommended_images[0],
                                                    recommended_image_2=recommended_images[1],
                                                    recommended_image_3=recommended_images[2],
                                                    label_korean=clothing_label_korean,
                                                    color_korean=color_label_korean,
                                                    _external=True)

            return jsonify(response_data)
        else:
            app.logger.error('허용되지 않는 파일 형식입니다.')
            return jsonify({'success': False, 'error': '허용되지 않는 파일 형식입니다.'}), 400
    except Exception as e:
        app.logger.error(f"서버 내부 오류 발생: {e}")
        return jsonify({'success': False, 'error': f'서버 내부 오류입니다: {str(e)}'}), 500


@app.route('/result/top')
def result_top():
    combined_recommendation_1 = request.args.get('combined_recommendation_1')
    combined_recommendation_2 = request.args.get('combined_recommendation_2')
    combined_recommendation_3 = request.args.get('combined_recommendation_3')
    recommended_image_1 = request.args.get('recommended_image_1')
    recommended_image_2 = request.args.get('recommended_image_2')
    recommended_image_3 = request.args.get('recommended_image_3')
    
    # label_korean과 color_korean 값을 명확히 가져옵니다.
    label_korean = request.args.get('label_korean')
    color_korean = request.args.get('color_korean')

    return render_template('top_analyze.html', 
                           combined_recommendation_1=combined_recommendation_1,
                           combined_recommendation_2=combined_recommendation_2,
                           combined_recommendation_3=combined_recommendation_3,
                           recommended_image_1=recommended_image_1,
                           recommended_image_2=recommended_image_2,
                           recommended_image_3=recommended_image_3,
                           label_korean=label_korean,  # 템플릿에 전달
                           color_korean=color_korean)  # 템플릿에 전달

@app.route('/result/bottom')
def result_bottom():
    combined_recommendation_1 = request.args.get('combined_recommendation_1')
    combined_recommendation_2 = request.args.get('combined_recommendation_2')
    combined_recommendation_3 = request.args.get('combined_recommendation_3')
    recommended_image_1 = request.args.get('recommended_image_1')
    recommended_image_2 = request.args.get('recommended_image_2')
    recommended_image_3 = request.args.get('recommended_image_3')
    
    # label_korean과 color_korean 값을 명확히 가져옵니다.
    label_korean = request.args.get('label_korean')
    color_korean = request.args.get('color_korean')

    return render_template('bottom_analyze.html', 
                           combined_recommendation_1=combined_recommendation_1,
                           combined_recommendation_2=combined_recommendation_2,
                           combined_recommendation_3=combined_recommendation_3,
                           recommended_image_1=recommended_image_1,
                           recommended_image_2=recommended_image_2,
                           recommended_image_3=recommended_image_3,
                           label_korean=label_korean,  # 템플릿에 전달
                           color_korean=color_korean)  # 템플릿에 전달


if __name__ == '__main__':
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.run(debug=True)

