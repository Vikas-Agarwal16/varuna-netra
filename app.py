import os
import io
import csv
import math
import traceback
from datetime import datetime, timedelta

from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from sqlalchemy import func, or_

from sonar_engine import SonarAnomalyEngine
from database import db, init_db, save_survey, Survey, Anomaly

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
RESULTS_FOLDER = os.path.join(BASE_DIR, "static", "results")
MODEL_PATH = os.path.join(BASE_DIR, "models", "best.pt")
TEMPLATE_FOLDER = os.path.join(BASE_DIR, "templates")
STATIC_FOLDER = os.path.join(BASE_DIR, "static")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

ALLOWED_EXT = {"png", "jpg", "jpeg", "bmp", "tiff", "tif", "webp"}

app = Flask(__name__, template_folder=TEMPLATE_FOLDER, static_folder=STATIC_FOLDER)
CORS(app)
app.config["MAX_CONTENT_LENGTH"] = 4 * 1024 * 1024

init_db(app)

engine = None
engine_error = None
try:
    print("[INIT] Loading model from:", MODEL_PATH)
    engine = SonarAnomalyEngine(model_path=MODEL_PATH, conf_thresh=0.40)
    print("[INIT] Engine ready")
except Exception as e:
    engine_error = str(e)
    print("[INIT ERROR]", engine_error)
    traceback.print_exc()


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT


def clamp(v, lo, hi):
    return min(max(v, lo), hi)


def priority_score(d):
    hazard_weight = {
        "CRITICAL": 1.00,
        "HIGH": 0.82,
        "MEDIUM": 0.60,
        "LOW": 0.35,
    }.get(str(d.get("hazard_level", "LOW")).upper(), 0.35)
    conf = clamp(float(d.get("confidence", 0)), 0, 100) / 100
    area = max(float(d.get("area_sq_m") or 0), 0)
    size_factor = min(math.log1p(area) / math.log(11), 1.0) if area else 0.0
    score = (0.55 * hazard_weight + 0.30 * conf + 0.15 * size_factor) * 100
    return round(clamp(score, 0, 100), 1)


def enrich_detection(d):
    x = dict(d)
    x["priority_score"] = priority_score(x)
    # This is an explainability heuristic, not a trained shadow classifier.
    x["shadow_evidence"] = round(clamp(
        35 + float(x.get("confidence", 0)) * 0.55 +
        min(float(x.get("area_sq_m") or 0) * 2, 10), 0, 100
    ), 1)
    x["explanation"] = [
        "Bounding-box shape matches a trained detection class",
        "Confidence exceeds the configured inference threshold",
        "Geospatial position calculated from sonar range and image position",
    ]
    return x


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/health")
def health():
    return jsonify({
        "status": "online" if engine is not None else "model_failed",
        "model_loaded": engine is not None,
        "model_path": MODEL_PATH,
        "model_exists": os.path.exists(MODEL_PATH),
        "error": engine_error,
        "accuracy": "92.60% mAP@50",
        "speed": "18.5ms inference",
    })


@app.route("/api/detect", methods=["POST"])
def detect():
    try:
        if engine is None:
            return jsonify({"success": False, "error": f"Model not loaded: {engine_error}"}), 500

        if "file" not in request.files:
            return jsonify({"success": False, "error": "No sonar image uploaded"}), 400

        file = request.files["file"]
        if not file or not file.filename.strip():
            return jsonify({"success": False, "error": "Empty filename"}), 400
        if not allowed_file(file.filename):
            return jsonify({"success": False, "error": "Unsupported image format"}), 400

        conf_thresh = clamp(float(request.form.get("confidence", 0.40)), 0.05, 0.95)
        range_meters = clamp(float(request.form.get("range", 50.0)), 1.0, 1000.0)
        base_lat = float(request.form.get("latitude", 18.9438))
        base_lon = float(request.form.get("longitude", 72.8360))
        survey_name = (request.form.get("survey_name") or "").strip()

        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        safe_name = os.path.basename(file.filename)
        upload_name = f"{ts}_{safe_name}"
        upload_path = os.path.join(UPLOAD_FOLDER, upload_name)
        file.save(upload_path)

        output_name = f"result_{upload_name}"
        output_path = os.path.join(RESULTS_FOLDER, output_name)

        print(f"[DETECT] file={upload_path}")
        print(f"[DETECT] conf={conf_thresh}, range={range_meters}, lat={base_lat}, lon={base_lon}")

        _, raw_detections = engine.process_sonar_image(
            img_path=upload_path,
            output_path=output_path,
            base_lat=base_lat,
            base_lon=base_lon,
            range_meters=range_meters,
            conf_override=conf_thresh,
        )

        detections = [enrich_detection(d) for d in raw_detections]
        summary = engine.generate_summary(detections)

        survey = save_survey(
            filename=safe_name,
            raw_path=f"/static/uploads/{upload_name}",
            result_path=f"/static/results/{output_name}",
            base_lat=base_lat,
            base_lon=base_lon,
            range_meters=range_meters,
            conf_thresh=conf_thresh,
            detections=detections,
            summary=summary,
            survey_name=survey_name or None,
        )

        return jsonify({
            "success": True,
            "survey_id": survey.id,
            "survey_name": survey.survey_name,
            "raw_image": f"/static/uploads/{upload_name}",
            "result_image": f"/static/results/{output_name}",
            "detections": detections,
            "summary": summary,
            "processing": {
                "noise_filter": "Bilateral acoustic noise reduction",
                "detector": "YOLO",
                "geolocation": "Range-based image-to-coordinate projection",
                "explainability": "Heuristic evidence layer",
            },
        })

    except Exception as e:
        print("[DETECT ERROR]", str(e))
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/surveys", methods=["GET"])
def list_surveys():
    page = max(int(request.args.get("page", 1)), 1)
    per_page = min(max(int(request.args.get("per_page", 20)), 1), 100)
    q = (request.args.get("q") or "").strip()

    query = Survey.query
    if q:
        like = f"%{q}%"
        query = query.filter(
            or_(
                Survey.survey_name.ilike(like),
                Survey.original_filename.ilike(like)
            )
        )

    total = query.count()
    surveys = query.order_by(Survey.created_at.desc()).offset(
        (page - 1) * per_page
    ).limit(per_page).all()

    return jsonify({
        "success": True,
        "page": page,
        "per_page": per_page,
        "total": total,
        "surveys": [s.to_dict() for s in surveys],
    })


@app.route("/api/surveys/<int:survey_id>")
def get_survey(survey_id):
    survey = db.session.get(Survey, survey_id)
    if survey is None:
        return jsonify({"success": False, "error": "Survey not found"}), 404

    data = survey.to_dict(include_anomalies=True)
    data["detections"] = [enrich_detection(d) for d in data["detections"]]
    return jsonify({"success": True, "survey": data})


@app.route("/api/surveys/<int:survey_id>", methods=["DELETE"])
def delete_survey(survey_id):
    survey = db.session.get(Survey, survey_id)
    if survey is None:
        return jsonify({"success": False, "error": "Survey not found"}), 404
    db.session.delete(survey)
    db.session.commit()
    return jsonify({"success": True, "deleted": survey_id})


@app.route("/api/anomalies/<int:anomaly_id>/review", methods=["POST"])
def review_anomaly(anomaly_id):
    anomaly = db.session.get(Anomaly, anomaly_id)
    if anomaly is None:
        return jsonify({"success": False, "error": "Anomaly not found"}), 404

    payload = request.get_json(silent=True) or {}
    decision = payload.get("decision") or payload.get("status")
    reviewer = payload.get("reviewed_by", "Operator")

    if decision not in ("approved", "rejected"):
        return jsonify({"success": False, "error": "decision must be approved or rejected"}), 400

    anomaly.review_status = decision
    anomaly.reviewed_by = reviewer
    anomaly.reviewed_at = datetime.now()

    survey = db.session.get(Survey, anomaly.survey_id)
    if survey:
        pending = Anomaly.query.filter_by(
            survey_id=survey.id, review_status="pending"
        ).count()
        survey.status = "reviewed" if pending == 0 else "pending"

    db.session.commit()
    return jsonify({"success": True, "anomaly": enrich_detection(anomaly.to_dict())})


@app.route("/api/dashboard/recent")
def recent_detections():
    limit = min(max(int(request.args.get("limit", 8)), 1), 100)
    rows = (
        db.session.query(Anomaly, Survey.created_at, Survey.original_filename)
        .join(Survey, Anomaly.survey_id == Survey.id)
        .order_by(Survey.created_at.desc(), Anomaly.id.desc())
        .limit(limit)
        .all()
    )

    results = []
    for anomaly, created_at, filename in rows:
        d = enrich_detection(anomaly.to_dict())
        d["survey_filename"] = filename
        d["detected_at"] = created_at.isoformat()
        results.append(d)

    return jsonify({"success": True, "anomalies": results, "recent": results})


@app.route("/api/dashboard/stats")
def dashboard_stats():
    total_surveys = Survey.query.count()
    total_anomalies = Anomaly.query.count()
    critical = Anomaly.query.filter_by(hazard_level="CRITICAL").count()

    avg_conf = db.session.query(func.avg(Anomaly.confidence)).scalar() or 0
    daily = (
        db.session.query(func.date(Survey.created_at), func.sum(Survey.total_detections))
        .group_by(func.date(Survey.created_at))
        .order_by(func.date(Survey.created_at))
        .all()
    )

    return jsonify({
        "success": True,
        "total_surveys": total_surveys,
        "total_anomalies": total_anomalies,
        "avg_confidence": round(float(avg_conf), 2),
        "critical": critical,
        "hazard_breakdown": dict(
            db.session.query(Anomaly.hazard_level, func.count(Anomaly.id))
            .group_by(Anomaly.hazard_level).all()
        ),
        "class_breakdown": dict(
            db.session.query(Anomaly.classification, func.count(Anomaly.id))
            .group_by(Anomaly.classification)
            .order_by(func.count(Anomaly.id).desc()).all()
        ),
        "review_breakdown": dict(
            db.session.query(Anomaly.review_status, func.count(Anomaly.id))
            .group_by(Anomaly.review_status).all()
        ),
        "daily_trend": [{"date": str(d), "detections": int(c or 0)} for d, c in daily],
    })


@app.route("/api/analytics")
def analytics():
    total = Anomaly.query.count()
    avg = db.session.query(func.avg(Anomaly.confidence)).scalar() or 0
    area = db.session.query(func.sum(Anomaly.area_sq_m)).scalar() or 0

    hazard = dict(
        db.session.query(Anomaly.hazard_level, func.count(Anomaly.id))
        .group_by(Anomaly.hazard_level).all()
    )
    review = dict(
        db.session.query(Anomaly.review_status, func.count(Anomaly.id))
        .group_by(Anomaly.review_status).all()
    )
    classes = dict(
        db.session.query(Anomaly.classification, func.count(Anomaly.id))
        .group_by(Anomaly.classification)
        .order_by(func.count(Anomaly.id).desc()).all()
    )

    return jsonify({
        "success": True,
        "total": total,
        "avg_confidence": round(float(avg), 2),
        "total_area": round(float(area), 2),
        "hazard": hazard,
        "review": review,
        "classes": classes,
    })


@app.route("/api/map/anomalies")
def map_anomalies():
    query = Anomaly.query

    hazard = request.args.get("hazard_level")
    classification = request.args.get("classification")
    review_status = request.args.get("review_status")
    min_confidence = float(request.args.get("min_confidence", 0) or 0)

    if hazard and hazard.upper() != "ALL":
        query = query.filter(Anomaly.hazard_level == hazard.upper())
    if classification and classification != "ALL":
        query = query.filter(Anomaly.classification == classification)
    if review_status and review_status != "ALL":
        query = query.filter(Anomaly.review_status == review_status)
    if min_confidence > 0:
        query = query.filter(Anomaly.confidence >= min_confidence)

    anomalies = [enrich_detection(a.to_dict()) for a in query.all()]
    return jsonify({"success": True, "count": len(anomalies), "anomalies": anomalies})


@app.route("/api/anomalies/<int:anomaly_id>/context")
def anomaly_context(anomaly_id):
    target = db.session.get(Anomaly, anomaly_id)
    if target is None:
        return jsonify({"success": False, "error": "Anomaly not found"}), 404

    # Spatial history: same class within a small coordinate window.
    delta = 0.00025
    nearby = Anomaly.query.filter(
        Anomaly.classification == target.classification,
        Anomaly.latitude.between(target.latitude - delta, target.latitude + delta),
        Anomaly.longitude.between(target.longitude - delta, target.longitude + delta),
    ).all()

    return jsonify({
        "success": True,
        "anomaly": enrich_detection(target.to_dict()),
        "history_count": len(nearby),
        "history": [
            {
                "survey_id": a.survey_id,
                "anomaly_id": a.anomaly_id,
                "latitude": a.latitude,
                "longitude": a.longitude,
                "hazard_level": a.hazard_level,
            }
            for a in nearby
        ],
    })


@app.route("/api/hotspots")
def hotspots():
    rows = Anomaly.query.all()
    cells = {}

    # Approximate display grid; not a scientific density estimator.
    for a in rows:
        key = (round(a.latitude, 3), round(a.longitude, 3))
        cell = cells.setdefault(key, {
            "latitude": key[0],
            "longitude": key[1],
            "count": 0,
            "critical": 0,
            "high": 0,
            "area": 0.0,
        })
        cell["count"] += 1
        cell["area"] += float(a.area_sq_m or 0)
        if a.hazard_level == "CRITICAL":
            cell["critical"] += 1
        if a.hazard_level == "HIGH":
            cell["high"] += 1

    points = list(cells.values())
    for p in points:
        p["priority"] = round(clamp(
            p["count"] * 12 + p["critical"] * 20 + p["high"] * 8 +
            min(p["area"], 40), 0, 100
        ), 1)

    points.sort(key=lambda x: x["priority"], reverse=True)
    return jsonify({"success": True, "hotspots": points[:20]})


@app.route("/api/download/json", methods=["POST"])
def download_json():
    data = request.get_json(silent=True) or {}
    output = {
        "system": "VARUN NETRA — Underwater Intelligence System",
        "problem_statement": "SIH 2026 PS-57",
        "generated_at": datetime.now().isoformat(),
        "summary": data.get("summary", {}),
        "detections": [enrich_detection(d) for d in data.get("detections", [])],
    }
    return jsonify(output)


@app.route("/api/download/csv", methods=["POST"])
def download_csv():
    data = request.get_json(silent=True) or {}
    detections = data.get("detections", [])

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow([
        "Anomaly ID", "Class", "Confidence (%)", "Hazard Level",
        "Priority Score", "Shadow Evidence (%)",
        "Latitude", "Longitude", "Width (m)", "Length (m)", "Area (m2)",
        "Review Status"
    ])

    for d in detections:
        x = enrich_detection(d)
        writer.writerow([
            x.get("anomaly_id"), x.get("classification"), x.get("confidence"),
            x.get("hazard_level"), x.get("priority_score"),
            x.get("shadow_evidence"), x.get("latitude"), x.get("longitude"),
            x.get("width_m"), x.get("length_m"), x.get("area_sq_m"),
            x.get("review_status", "pending")
        ])

    return output.getvalue(), 200, {
        "Content-Type": "text/csv",
        "Content-Disposition": "attachment; filename=varun_netra_anomaly_report.csv",
    }


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(
        host="0.0.0.0",
        port=port,
        debug=False,
        use_reloader=False
    )