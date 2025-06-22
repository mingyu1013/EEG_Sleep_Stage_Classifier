EEG 기반 수면 단계 분류 웹 애플리케이션
🔍 프로젝트 개요
이 프로젝트는 EEG(뇌파) 데이터를 이용해 수면 단계를 분류하고, Flask 기반 웹 애플리케이션을 통해 결과를 시각화하여 확인할 수 있도록 구성되었습니다.

📁 프로젝트 구조
폴더명	설명
flask_app/	Flask 웹 애플리케이션 구현
model/	CNN+LSTM 구조의 학습된 모델 및 실행 스크립트
preprocessing/	EDF EEG 데이터 전처리 코드 및 결과 파일

⚙️ 실행 환경
Python 3.10 이상 권장

가상환경 설정 (Windows 기준):

bash
복사
편집
python -m venv .venv
.venv\Scripts\activate
pip install -r flask_app/requirements.txt
📦 모델과 데이터
모델: 학습된 모델 cnn_lstm_best.pth 는 GitHub 저장소에 포함되어 있음

데이터: preprocessed_gpu.npz (용량 문제로 미포함)

전처리 직접 실행:
preprocessing/Preprocessing.py 실행 필요

✅ 전처리 파일 다운로드:
Google Drive - preprocessed_gpu.npz
https://drive.google.com/file/d/1_Xkop2rDumjUdNJA7ginP9pijYsotlWx/view?usp=drive_link
✅ 원본 공개 데이터셋:
PhysioNet Sleep-EDF
https://physionet.org/content/sleep-edfx/1.0.0/

🚀 실행 방법
bash
복사
편집
cd flask_app
python app.py
실행 후 웹 브라우저에서 아래 주소로 접속:

📍 http://127.0.0.1:5000

💡 사용 예시
EEG 파일 업로드 또는 예시 데이터 선택

수면 단계별 예측 결과 확인

수면 패턴 그래프 및 표 형태로 시각화

🧯 오류 발생 시 대처
문제	해결 방법
ModuleNotFoundError	pip install -r requirements.txt 실행
preprocessed_gpu.npz 없음	경로 확인 또는 다운로드 링크 활용

📚 참고
데이터 출처: Sleep-EDF Expanded Dataset (PhysioNet)

모델 구조: CNN + LSTM (PyTorch 기반)
