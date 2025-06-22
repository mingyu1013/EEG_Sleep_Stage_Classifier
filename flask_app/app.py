from flask import Flask, render_template, request, redirect, url_for, session, flash
from models import Report
import json
from datetime import datetime, timedelta
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import matplotlib
matplotlib.use('Agg')
import mne
import numpy as np
import torch
from model_architecture import CNN_LSTM
from feedback import generate_feedback
from collections import Counter
import matplotlib.pyplot as plt
from werkzeug.utils import secure_filename

from flask_login import LoginManager, login_user, login_required, logout_user, current_user
from user_model import db, User

import plotly.express as px
import pandas as pd
import io
import base64

app = Flask(__name__)
app.secret_key = 'your_secret_key'
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
db.init_app(app)

login_manager = LoginManager()
login_manager.login_view = 'login'
login_manager.init_app(app)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

model = CNN_LSTM()
model.load_state_dict(torch.load("cnn_lstm_best.pth", map_location=torch.device("cpu")))
model.eval()

with app.app_context():
    db.create_all()

def process_edf(file_path, fs=100, win_sec=30, stride_sec=30):
    raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)
    raw.pick_channels(["EEG Pz-Oz"])
    raw.filter(0.5, 30.0, fir_design="firwin", verbose=False)
    raw.resample(fs, npad="auto")

    sig = raw.get_data()[0]
    win_len = int(win_sec * fs)
    stride = int(stride_sec * fs)

    def sliding_windows(x, win_len, stride):
        n = x.shape[0]
        num = (n - win_len) // stride + 1
        shape = (num, win_len)
        strides = (x.strides[0] * stride, x.strides[0])
        return np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)

    segments = sliding_windows(sig, win_len, stride)
    if len(segments) == 0:
        return np.empty((0, win_len), dtype=np.float32)
    segments = (segments - segments.mean(axis=1, keepdims=True)) / (segments.std(axis=1, keepdims=True) + 1e-6)
    return segments.astype(np.float32)

def save_stage_pie(preds, save_path="static/percent_pie.png"):
    import matplotlib.pyplot as plt
    import matplotlib.font_manager as fm

    font_path = "C:/Windows/Fonts/malgun.ttf"
    fontprop = fm.FontProperties(fname=font_path)
    plt.rcParams['font.family'] = fontprop.get_name()
    plt.rcParams['axes.unicode_minus'] = False

    stage_names = ['Wake', 'N1', 'N2', 'N3', 'REM']
    counts = Counter(preds)

    labels = []
    sizes = []
    for i in range(1, 5):
        if counts.get(i, 0) > 0:
            labels.append(stage_names[i])
            sizes.append(counts[i])

    if not sizes:
        labels = ['수면 없음']
        sizes = [1]

    zipped = list(zip(labels, sizes))
    zipped.sort(key=lambda x: x[1])
    labels, sizes = zip(*zipped)

    total = sum(sizes)
    pct_values = [round((s / total) * 100, 1) for s in sizes]

    colors = ['#FFC1C1', '#E2D1F9', '#FAEDCB', '#BEE1E6'][:len(sizes)]

    plt.figure(figsize=(5, 5))
    plt.pie(
        sizes,
        labels=None,
        autopct='%1.1f%%',
        startangle=180,
        counterclock=False,
        colors=colors,
        wedgeprops={'edgecolor': 'white', 'linewidth': 2},
        textprops={'fontsize': 12}
    )

    plt.tight_layout()
    plt.savefig(save_path, transparent=True)
    plt.close()

    return list(labels), pct_values

@app.route("/score_chart")
@login_required
def score_chart():
    reports = Report.query.filter_by(user_id=current_user.id).order_by(Report.timestamp.asc()).all()
    if not reports:
        return render_template("score_chart.html", chart_html=None)

    df = pd.DataFrame({
        "분석 번호": list(range(1, len(reports)+1)),
        "수면 점수": [r.sleep_score for r in reports]
    })

    fig = px.scatter(df, x="분석 번호", y="수면 점수",
                     title="수면 점수 변화",
                     labels={"수면 점수": "수면 질 ( 0 ~ 100점 )"},
                     range_y=[0, 100])
    fig.update_traces(marker=dict(size=10, color="#1e40af"))
    fig.update_layout(
        yaxis=dict(tickmode='array', tickvals=[0, 50, 100], ticktext=["나쁨", "보통", "좋음"]),
        xaxis=dict(title="분석 횟수", tickmode="linear", dtick=1)
    )

    graph_html = fig.to_html(full_html=False, config={"staticPlot": True})
    return render_template("score_chart.html", chart_html=graph_html)

@app.route("/", methods=["GET", "POST"])
@login_required
def index():
    if request.method == "POST":
        file = request.files["eeg_file"]
        if file and file.filename.endswith(".edf"):
            filename = secure_filename(file.filename)
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            file.save(filepath)
            try:
                inputs = process_edf(filepath)
                if len(inputs) == 0:
                    return render_template("index.html", analyzing=False, feedback_list=["데이터가 너무 짧습니다."])
                with torch.no_grad():
                    x = torch.tensor(inputs).float().unsqueeze(1)
                    preds = model(x).argmax(dim=1).numpy().tolist()
                    session['preds'] = preds
                    return redirect(url_for("result"))
            except Exception as e:
                print("🚨 예외:", e)
                return render_template("index.html", analyzing=False, feedback_list=[str(e)])
    return render_template("index.html", analyzing=False)

@app.route("/feedback")
@login_required
def feedback():
    major_stage_code = session.get("major_stage", "")
    print("📌 세션 major_stage 값:", repr(major_stage_code))  # 로그 출력

    return render_template("feedback.html", major_stage=major_stage_code)

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        email = request.form['email']
        password = request.form['password']
        if User.query.filter_by(email=email).first():
            flash("이미 등록된 이메일입니다.")
            return redirect("/register")
        new_user = User(email=email)
        new_user.set_password(password)
        db.session.add(new_user)
        db.session.commit()
        return redirect("/register-complete")
    return render_template("register.html")

@app.route("/register-complete")
def register_complete():
    return render_template("register_complete.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form['email']
        password = request.form['password']
        user = User.query.filter_by(email=email).first()
        if user and user.check_password(password):
            login_user(user)
            return redirect("/")
        flash("이메일 혹은 비밀번호를 잘못 입력하셨거나 등록되지 않은 이메일입니다.")
    return render_template("login.html")

@app.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect("/")

@app.route("/find-id", methods=["GET", "POST"])
def find_id():
    if request.method == "POST":
        flash("기능 준비 중입니다.")
    return render_template("find_id.html")

@app.route("/find-password", methods=["GET", "POST"])
def find_password():
    if request.method == "POST":
        flash("기능 준비 중입니다.")
    return render_template("find_password.html")

stage_label_to_code = {
    "얕은 수면(N1)": "N1",
    "중간 수면(N2)": "N2",
    "깊은 수면(N3)": "N3",
    "REM 수면": "REM"
}

SLEEP_STAGE_DESCRIPTIONS = {
    "N1": "잠에 들기 시작하면 나타나는 가장 얕은 단계이며 쉽게 깨어날 수 있어요.",
    "N2": "얕은 잠에서 깊은 잠으로 넘어가는 중간 단계로 근육이 이완되고 심박수와 체온이 하강되어요. 성인 수면의 가장 많은 부분을 차지하고 있습니다.",
    "N3": "깊은 잠의 단계로 신체적 휴식단계예요. 성장 호르몬 분비, 세포 복구 등 신체 재생이 활발하게 움직이는 가장 중요한 단계랍니다.",
    "REM": "꿈을 꾸는 단계로 뇌가 활발하게 움직이며 근육이 일시적으로 마비 돼요."
}

@app.route("/result")
@login_required
def result():
    preds = session.get('preds')
    if preds is None:
        return redirect(url_for("index"))

    labels, pcts = save_stage_pie(preds)
    pct_dict = {label: pct for label, pct in zip(labels, pcts)}

    n1_pct = pct_dict.get("N1", 0)
    n2_pct = pct_dict.get("N2", 0)
    n3_pct = pct_dict.get("N3", 0)
    rem_pct = pct_dict.get("REM", 0)

    counts = Counter(preds)
    total = len(preds)
    most_common_list = counts.most_common()
    non_wake = [item for item in most_common_list if item[0] != 0]
    most_common_index = non_wake[0][0] if non_wake else 0

    # ✅ 코드값 기준 major_stage (예: "N1", "N3" 등)
    stage_codes = ['Wake', 'N1', 'N2', 'N3', 'REM']
    stage_labels_kor = {
        'Wake': '깨어 있음(Wake)',
        'N1': '얕은 수면(N1)',
        'N2': '중간 수면(N2)',
        'N3': '깊은 수면(N3)',
        'REM': 'REM 수면'
    }

    major_stage_code = stage_codes[most_common_index]
    major_stage_label = stage_labels_kor[major_stage_code]

    # ✅ 세션에 코드값 저장 (피드백 판별용)
    session["major_stage"] = major_stage_code

    actual_ratio = {
        1: n1_pct / 100,
        2: n2_pct / 100,
        3: n3_pct / 100,
        4: rem_pct / 100
    }
    ideal_ratio = {1: 0.04, 2: 0.47, 3: 0.12, 4: 0.22}

    partial_scores = [
        max(0, (1 - abs(actual_ratio.get(stage, 0) - ideal_ratio[stage]) / ideal_ratio[stage])) * 100
        for stage in ideal_ratio
    ]
    sleep_score = int(sum(partial_scores) / len(partial_scores))

    gap_info = {
        stage: round((actual_ratio[stage] - ideal_ratio[stage]) * 100, 1)
        for stage in ideal_ratio
    }

    report = Report(
        user_id=current_user.id,
        preds_json=json.dumps(preds),
        sleep_score=sleep_score,
        major_stage=major_stage_label  # ✅ DB에는 한글라벨 저장
    )
    db.session.add(report)
    db.session.commit()

    total_minutes = len(preds) * 0.5
    hours = int(total_minutes // 60)
    minutes = int(total_minutes % 60)
    total_sleep_time = f"{hours}시간 {minutes}분"

    stage_pct_dict = {
        'N1': n1_pct,
        'N2': n2_pct,
        'N3': n3_pct,
        'REM': rem_pct
    }
    highest_stage = max(stage_pct_dict, key=stage_pct_dict.get)
    character_img = f"{highest_stage}.png"

    stage_description = SLEEP_STAGE_DESCRIPTIONS.get(major_stage_code, "")

    return render_template("result.html",
                           major_stage=major_stage_code, 
                           sleep_score=sleep_score,
                           n1_pct=n1_pct,
                           n2_pct=n2_pct,
                           n3_pct=n3_pct,
                           rem_pct=rem_pct,
                           total_sleep_time=total_sleep_time,
                           gap_n1=gap_info[1],
                           gap_n2=gap_info[2],
                           gap_n3=gap_info[3],
                           gap_rem=gap_info[4],
                           character_img=character_img,
                           stage_description=stage_description)


@app.route("/reports")
@login_required
def reports():
    user_reports = Report.query.filter_by(user_id=current_user.id).order_by(Report.timestamp.desc()).all()
    return render_template("reports.html", reports=user_reports, timedelta=timedelta)

@app.route("/report/<int:report_id>")
@login_required
def report_detail(report_id):
    report = Report.query.get_or_404(report_id)
    if report.user_id != current_user.id:
        flash("접근 권한이 없습니다.")
        return redirect(url_for("reports"))

    preds = json.loads(report.preds_json)

    labels, pcts = save_stage_pie(preds)
    pct_dict = {label: pct for label, pct in zip(labels, pcts)}
    n1_pct = pct_dict.get("N1", 0)
    n2_pct = pct_dict.get("N2", 0)
    n3_pct = pct_dict.get("N3", 0)
    rem_pct = pct_dict.get("REM", 0)

    actual_ratio = {
        1: n1_pct / 100,
        2: n2_pct / 100,
        3: n3_pct / 100,
        4: rem_pct / 100
    }
    ideal_ratio = {1: 0.04, 2: 0.47, 3: 0.12, 4: 0.22}

    gap_info = {
        stage: round((actual_ratio[stage] - ideal_ratio[stage]) * 100, 1)
        for stage in ideal_ratio
    }

    stage_pct_dict = {
        'N1': n1_pct,
        'N2': n2_pct,
        'N3': n3_pct,
        'REM': rem_pct
    }
    highest_stage = max(stage_pct_dict, key=stage_pct_dict.get)
    character_img = f"{highest_stage}.png"

    total_minutes = len(preds) * 0.5
    hours = int(total_minutes // 60)
    minutes = int(total_minutes % 60)
    total_sleep_time = f"{hours}시간 {minutes}분"

    
    stage_code = stage_label_to_code.get(report.major_stage, None)
    session["major_stage"] = stage_code
    stage_description = SLEEP_STAGE_DESCRIPTIONS.get(stage_code, "")

    return render_template("result.html",
                           major_stage=report.major_stage,
                           sleep_score=report.sleep_score,
                           n1_pct=n1_pct,
                           n2_pct=n2_pct,
                           n3_pct=n3_pct,
                           rem_pct=rem_pct,
                           total_sleep_time=total_sleep_time,
                           gap_n1=gap_info[1],
                           gap_n2=gap_info[2],
                           gap_n3=gap_info[3],
                           gap_rem=gap_info[4],
                           character_img=character_img,
                           stage_description=stage_description)


if __name__ == "__main__":
    app.run(debug=True)