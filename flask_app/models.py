# models.py
from datetime import datetime
import json
from user_model import db  # ✅ 공용 db 인스턴스 사용

class Report(db.Model):
    __tablename__ = 'report'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    preds_json = db.Column(db.Text, nullable=False)
    sleep_score = db.Column(db.Integer, nullable=False)
    major_stage = db.Column(db.String(50), nullable=False)

    def get_preds(self):
        return json.loads(self.preds_json)
