"""
VARUN NETRA persistence layer.
Keeps the existing Survey and Anomaly schema compatible with sonar_data.db.
"""

import os
from datetime import datetime
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "sonar_data.db")


class Survey(db.Model):
    __tablename__ = "surveys"

    id = db.Column(db.Integer, primary_key=True)
    survey_name = db.Column(db.String(255), nullable=False)
    original_filename = db.Column(db.String(255), nullable=False)
    raw_image_path = db.Column(db.String(512), nullable=False)
    result_image_path = db.Column(db.String(512), nullable=False)

    base_lat = db.Column(db.Float, nullable=False)
    base_lon = db.Column(db.Float, nullable=False)
    range_meters = db.Column(db.Float, nullable=False)
    confidence_threshold = db.Column(db.Float, nullable=False)

    total_detections = db.Column(db.Integer, default=0)
    critical_count = db.Column(db.Integer, default=0)
    high_count = db.Column(db.Integer, default=0)
    medium_count = db.Column(db.Integer, default=0)
    low_count = db.Column(db.Integer, default=0)
    avg_confidence = db.Column(db.Float, default=0.0)
    total_area_sq_m = db.Column(db.Float, default=0.0)

    status = db.Column(db.String(20), default="pending")
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    anomalies = db.relationship(
        "Anomaly", backref="survey", cascade="all, delete-orphan", lazy=True
    )

    def to_dict(self, include_anomalies=False):
        data = {
            "id": self.id,
            "survey_name": self.survey_name,
            "original_filename": self.original_filename,
            "raw_image": self.raw_image_path,
            "result_image": self.result_image_path,
            "base_lat": self.base_lat,
            "base_lon": self.base_lon,
            "range_meters": self.range_meters,
            "confidence_threshold": self.confidence_threshold,
            "summary": {
                "total": self.total_detections,
                "critical": self.critical_count,
                "high": self.high_count,
                "medium": self.medium_count,
                "low": self.low_count,
                "avg_confidence": self.avg_confidence,
                "total_area": self.total_area_sq_m,
            },
            "status": self.status,
            "created_at": self.created_at.isoformat(),
        }
        if include_anomalies:
            data["detections"] = [a.to_dict() for a in self.anomalies]
        return data


class Anomaly(db.Model):
    __tablename__ = "anomalies"

    id = db.Column(db.Integer, primary_key=True)
    survey_id = db.Column(db.Integer, db.ForeignKey("surveys.id"), nullable=False)

    anomaly_id = db.Column(db.String(50), nullable=False)
    classification = db.Column(db.String(100), nullable=False)
    confidence = db.Column(db.Float, nullable=False)
    hazard_level = db.Column(db.String(20), nullable=False)

    latitude = db.Column(db.Float, nullable=False)
    longitude = db.Column(db.Float, nullable=False)
    width_m = db.Column(db.Float)
    length_m = db.Column(db.Float)
    area_sq_m = db.Column(db.Float)

    bbox_x1 = db.Column(db.Integer)
    bbox_y1 = db.Column(db.Integer)
    bbox_x2 = db.Column(db.Integer)
    bbox_y2 = db.Column(db.Integer)

    review_status = db.Column(db.String(20), default="pending")
    reviewed_by = db.Column(db.String(100), nullable=True)
    reviewed_at = db.Column(db.DateTime, nullable=True)

    def to_dict(self):
        return {
            "id": self.id,
            "survey_id": self.survey_id,
            "anomaly_id": self.anomaly_id,
            "classification": self.classification,
            "confidence": self.confidence,
            "hazard_level": self.hazard_level,
            "latitude": self.latitude,
            "longitude": self.longitude,
            "width_m": self.width_m,
            "length_m": self.length_m,
            "area_sq_m": self.area_sq_m,
            "bbox": [self.bbox_x1, self.bbox_y1, self.bbox_x2, self.bbox_y2],
            "review_status": self.review_status,
            "reviewed_by": self.reviewed_by,
            "reviewed_at": self.reviewed_at.isoformat() if self.reviewed_at else None,
        }


def init_db(app):
    app.config["SQLALCHEMY_DATABASE_URI"] = f"sqlite:///{DB_PATH}"
    app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
    db.init_app(app)
    with app.app_context():
        db.create_all()
    print(f"[DB] Ready at {DB_PATH}")


def save_survey(
    filename,
    raw_path,
    result_path,
    base_lat,
    base_lon,
    range_meters,
    conf_thresh,
    detections,
    summary,
    survey_name=None,
):
    survey = Survey(
        survey_name=survey_name or f"Survey {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')}",
        original_filename=filename,
        raw_image_path=raw_path,
        result_image_path=result_path,
        base_lat=base_lat,
        base_lon=base_lon,
        range_meters=range_meters,
        confidence_threshold=conf_thresh,
        total_detections=summary.get("total", 0),
        critical_count=summary.get("critical", 0),
        high_count=summary.get("high", 0),
        medium_count=summary.get("medium", 0),
        low_count=summary.get("low", 0),
        avg_confidence=summary.get("avg_confidence", 0.0),
        total_area_sq_m=summary.get("total_area", 0.0),
    )

    db.session.add(survey)
    db.session.flush()

    for d in detections:
        bbox = d.get("bbox", [None, None, None, None])
        db.session.add(Anomaly(
            survey_id=survey.id,
            anomaly_id=d.get("anomaly_id"),
            classification=d.get("classification"),
            confidence=d.get("confidence"),
            hazard_level=d.get("hazard_level"),
            latitude=d.get("latitude"),
            longitude=d.get("longitude"),
            width_m=d.get("width_m"),
            length_m=d.get("length_m"),
            area_sq_m=d.get("area_sq_m"),
            bbox_x1=bbox[0],
            bbox_y1=bbox[1],
            bbox_x2=bbox[2],
            bbox_y2=bbox[3],
        ))

    db.session.commit()
    return survey
