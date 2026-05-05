import os
from datetime import datetime, timezone
from pathlib import Path
import math
import re

import cv2
from flask import Flask, render_template, request, send_from_directory, session, url_for
from werkzeug.utils import secure_filename

from config_db import DB_CONFIG
from main import (
    estimate_pointing_multi_triangle,
    estimate_pointing_and_earth_coordinates,
    fov_diag_limit_deg,
    find_stars_advanced,
    generate_triangles_from_stars,
)
from matcher import StarMatcher

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp", "webp"}

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = str(UPLOAD_DIR)
app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024
app.secret_key = os.environ.get("STAR_UI_SECRET", "star-ui-dev-secret")
STAR_DB = None


def get_star_db():
    global STAR_DB
    if STAR_DB is None:
        STAR_DB = StarMatcher(db_config=DB_CONFIG)
    return STAR_DB


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def parse_observation_time(raw_value: str):
    if not raw_value:
        return None
    text = raw_value.strip()
    if not text:
        return None

    # Accept common user formats like "YYYY-MM-DD HH.MM.SS UTC+03:00".
    text = text.replace(" UTC", " ").replace("utc", " ")
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"(\d{2})\.(\d{2})\.(\d{2})(?=\s|$|[+-])", r"\1:\2:\3", text)

    dt = None
    try:
        dt = datetime.fromisoformat(text.replace(" ", "T", 1) if "T" not in text and " " in text else text)
    except ValueError:
        pass

    if dt is None:
        for fmt in ("%Y-%m-%d %H:%M:%S%z", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%dT%H:%M:%S"):
            try:
                dt = datetime.strptime(text, fmt)
                break
            except ValueError:
                continue

    if dt is None:
        return None

    if dt.tzinfo is None:
        return None
    return dt.astimezone(timezone.utc)


def draw_triangles_preview(image_path, stars, triangles, out_path):
    img = cv2.imread(str(image_path))
    if img is None:
        return False

    colors = [(28, 206, 255), (60, 255, 140), (255, 170, 42)]
    num_triangles_to_draw = min(3, len(triangles))

    for i in range(num_triangles_to_draw):
        triangle_ids = triangles[i]["star_ids"]
        color = colors[i]

        points = []
        for s in stars:
            if s.id in triangle_ids:
                points.append((int(s.center[0]), int(s.center[1])))

        if len(points) == 3:
            p1, p2, p3 = points[0], points[1], points[2]
            cv2.line(img, p1, p2, color, 2)
            cv2.line(img, p2, p3, color, 2)
            cv2.line(img, p3, p1, color, 2)
            cv2.circle(img, p1, 4, color, -1)
            cv2.circle(img, p2, 4, color, -1)
            cv2.circle(img, p3, 4, color, -1)

    cv2.imwrite(str(out_path), img)
    return True


def parse_float_with_default(raw_value: str, default: float) -> float:
    text = (raw_value or "").strip()
    if not text:
        return default
    try:
        return float(text)
    except ValueError:
        return default


def parse_int_with_default(raw_value: str, default: int) -> int:
    text = (raw_value or "").strip()
    if not text:
        return default
    try:
        return int(text)
    except ValueError:
        return default


def evaluate_config(star_db, stars, image_path, fov_x_deg, obs_time_utc, top_n, tolerance, fast_mode=False):
    """Evaluate triangle configuration without orientation parameter.
    
    Orientation is now handled automatically within matcher.py through dual-space
    KD-Tree queries, so it does not need to be a configuration parameter.
    """
    probe_img = cv2.imread(str(image_path))
    if probe_img is not None:
        img_h, img_w = probe_img.shape[:2]
        image_shape = probe_img.shape
        min_sep_px = max(10.0, min(24.0, 0.012 * img_w * (35.0 / max(10.0, fov_x_deg))))
    else:
        image_shape = None
        min_sep_px = 12.0

    if image_shape is None:
        probe_img = cv2.imread(str(image_path))
        if probe_img is None:
            return {
                "triangles": [],
                "consensus": {"used_triangles": 0, "matched_triangles": 0, "candidates": []},
                "coordinate_solution": None,
                "top_n": top_n,
                "tolerance": tolerance,
                "score": 1e9,
            }
        image_shape = probe_img.shape

    max_side_deg = fov_diag_limit_deg(fov_x_deg, image_shape)

    triangles = generate_triangles_from_stars(
        stars,
        image_shape,
        fov_x_deg,
        top_n=top_n,
        min_separation_px=min_sep_px,
    )
    triangles_eval = triangles[:120] if fast_mode else triangles[:220]

    consensus = star_db.find_consensus_matches(
        triangles_eval[: min(12 if fast_mode else 24, len(triangles_eval))],
        tolerance=tolerance,
        per_triangle_k=24 if fast_mode else 40,
        fov_x_deg=fov_x_deg,  # Use actual FOV, not hardcoded
        max_side_deg=max_side_deg,
    )

    top_conf = consensus["candidates"][0]["confidence"] if consensus["candidates"] else 0.0
    top_support = consensus["candidates"][0]["support_count"] if consensus["candidates"] else 0

    coordinate_solution = None
    if not fast_mode and top_support >= 2:
        coordinate_solution = estimate_pointing_multi_triangle(
            star_db=star_db,
            stars=stars,
            image_path=str(image_path),
            triangles=triangles_eval,
            fov_x_deg=fov_x_deg,  # Use actual FOV, not hardcoded
            observation_time_utc=obs_time_utc,
            tolerance=tolerance,
            per_triangle_k=12,
            max_fit_rms_deg=3.0,
            max_side_deg=max_side_deg,
        )

    if coordinate_solution is None and top_support >= 2:
        for tri in triangles_eval:
            # Fallback: try single triangle with actual FOV
            matches = star_db.find_best_match(
                tri["ratio1"],
                tri["ratio2"],
                tri["ratio3"],
                tolerance=tolerance,
                fov_x_deg=fov_x_deg,  # Use actual FOV, not hardcoded
                k=1,
                max_side_deg=max_side_deg,
            )
            if not matches:
                continue

            coordinate_solution = estimate_pointing_and_earth_coordinates(
                star_db=star_db,
                stars=stars,
                image_path=str(image_path),
                image_triangle=tri["star_ids"],
                hip_triangle=matches[0]["star_hips"],
                fov_x_deg=fov_x_deg,  # Use actual FOV, not hardcoded
                observation_time_utc=obs_time_utc,
            )
            if coordinate_solution is not None:
                break

    if coordinate_solution is None:
        for tri in triangles_eval:
            matches = star_db.find_best_match(
                tri["ratio1"],
                tri["ratio2"],
                tri["ratio3"],
                tolerance=1e9,
                fov_x_deg=360.0,
                k=1,
                max_side_deg=360.0,
            )
            if not matches:
                continue

            coordinate_solution = estimate_pointing_and_earth_coordinates(
                star_db=star_db,
                stars=stars,
                image_path=str(image_path),
                image_triangle=tri["star_ids"],
                hip_triangle=matches[0]["star_hips"],
                fov_x_deg=fov_x_deg,
                observation_time_utc=obs_time_utc,
                force=True,
            )
            if coordinate_solution is not None:
                break

    if coordinate_solution is None:
        coordinate_solution = {
            "ra_deg": 0.0,
            "dec_deg": 0.0,
            "lat_deg": 0.0,
            "lon_deg": 0.0,
            "pointing_lat_deg": 0.0,
            "pointing_lon_deg": 0.0,
            "r_eci": [0.0, 0.0, 1.0],
            "fit_rms_deg": 999.0,
            "fit_max_deg": 999.0,
            "triangle_pair_rms_deg": 999.0,
            "timestamp_utc": obs_time_utc.isoformat(),
            "used_hips": (),
            "used_hypotheses": 0,
            "total_hypotheses": 0,
            "max_triplet_support": 0,
            "best_triangle_id": "",
        }

    if coordinate_solution is None:
        score = (1.0 - top_conf) if fast_mode else (1e9 - top_conf)
    else:
        fit = float(coordinate_solution.get("fit_rms_deg", 99.0))
        pair = float(coordinate_solution.get("triangle_pair_rms_deg", fit))
        spread = float(coordinate_solution.get("spread_deg", fit))
        hypotheses = float(coordinate_solution.get("used_hypotheses", 1.0))
        score = (
            fit
            + 0.75 * pair
            + 0.40 * spread
            + (1.0 - top_conf) * 1.45
            - 0.18 * min(10.0, hypotheses)
        )

    return {
        "triangles": triangles_eval,
        "consensus": consensus,
        "coordinate_solution": coordinate_solution,
        "top_n": top_n,
        "tolerance": tolerance,
        "score": score,
    }





@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    error = None
    form_values = {
        "fov_x_deg": session.get("fov_x_deg", "60.0"),
        "top_n": session.get("top_n", "8"),
        "tolerance": session.get("tolerance", "0.015"),
        "observation_time_utc": session.get("observation_time_utc", ""),
        "auto_tune": session.get("auto_tune", "on"),
    }
    last_image_filename = session.get("last_image_filename", "")

    if request.method == "POST":
        uploaded = request.files.get("image")
        form_values = {
            "fov_x_deg": (request.form.get("fov_x_deg", "60.0") or "").strip(),
            "top_n": (request.form.get("top_n", "8") or "").strip(),
            "tolerance": (request.form.get("tolerance", "0.015") or "").strip(),
            "observation_time_utc": (request.form.get("observation_time_utc", "") or "").strip(),
            "auto_tune": "on" if request.form.get("auto_tune") else "",
        }

        session.update(form_values)

        fov_x_deg = parse_float_with_default(form_values["fov_x_deg"], 60.0)
        top_n = max(3, min(30, parse_int_with_default(form_values["top_n"], 8)))
        tolerance = max(0.001, min(0.1, parse_float_with_default(form_values["tolerance"], 0.015)))
        auto_tune = form_values["auto_tune"] == "on"
        obs_time_utc = parse_observation_time(form_values["observation_time_utc"])

        if not form_values["observation_time_utc"]:
            error = "Вкажіть час знімка та UTC-офсет (напр. 2026-04-26 21:15:00 +03:00)."
            return render_template(
                "index.html",
                result=result,
                error=error,
                form_values=form_values,
                last_image_filename=last_image_filename,
            )
        if obs_time_utc is None:
            error = "Невірний формат часу або відсутній UTC-офсет (напр. +03:00)."
            return render_template(
                "index.html",
                result=result,
                error=error,
                form_values=form_values,
                last_image_filename=last_image_filename,
            )

        image_filename = None
        image_path = None

        if uploaded is not None and uploaded.filename != "":
            if not allowed_file(uploaded.filename):
                error = "Підтримуються лише PNG, JPG, JPEG, BMP, WEBP."
                return render_template(
                    "index.html",
                    result=result,
                    error=error,
                    form_values=form_values,
                    last_image_filename=last_image_filename,
                )

            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
            safe_name = secure_filename(uploaded.filename)
            image_filename = f"{timestamp}_{safe_name}"
            image_path = UPLOAD_DIR / image_filename
            uploaded.save(image_path)
            session["last_image_filename"] = image_filename
            last_image_filename = image_filename
        else:
            image_filename = session.get("last_image_filename")
            if image_filename:
                image_path = UPLOAD_DIR / image_filename

        if image_path is None or not image_path.exists():
            error = "Завантажте фото один раз. Далі його можна не перевантажувати."
            return render_template(
                "index.html",
                result=result,
                error=error,
                form_values=form_values,
                last_image_filename=last_image_filename,
            )

        star_db = None
        try:
            stars = find_stars_advanced(str(image_path))

            star_db = get_star_db()

            best_eval = evaluate_config(
                star_db=star_db,
                stars=stars,
                image_path=image_path,
                fov_x_deg=fov_x_deg,
                obs_time_utc=obs_time_utc,
                top_n=top_n,
                tolerance=tolerance,
                fast_mode=auto_tune,
            )

            if auto_tune:
                top_candidates = sorted({max(6, min(24, t)) for t in [top_n - 4, top_n - 2, top_n, top_n + 2, top_n + 4, top_n + 8, top_n + 12]})
                tol_candidates = sorted({max(0.006, min(0.06, v)) for v in [tolerance * 0.60, tolerance * 0.80, tolerance, tolerance * 1.20, tolerance * 1.50, tolerance * 1.80, tolerance * 2.20]})

                for tn in top_candidates:
                    for tol in tol_candidates:
                        candidate = evaluate_config(
                            star_db=star_db,
                            stars=stars,
                            image_path=image_path,
                            fov_x_deg=fov_x_deg,
                            obs_time_utc=obs_time_utc,
                            top_n=tn,
                            tolerance=tol,
                            fast_mode=True,
                        )
                        if candidate["score"] < best_eval["score"]:
                            best_eval = candidate

                best_eval = evaluate_config(
                    star_db=star_db,
                    stars=stars,
                    image_path=image_path,
                    fov_x_deg=fov_x_deg,
                    obs_time_utc=obs_time_utc,
                    top_n=best_eval["top_n"],
                    tolerance=best_eval["tolerance"],
                    fast_mode=False,
                )

                top_n = best_eval["top_n"]
                tolerance = best_eval["tolerance"]
                form_values["top_n"] = str(top_n)
                form_values["tolerance"] = f"{tolerance:.5f}".rstrip("0").rstrip(".")
                session["top_n"] = form_values["top_n"]
                session["tolerance"] = form_values["tolerance"]

            triangles = best_eval["triangles"]
            consensus = best_eval["consensus"]
            coordinate_solution = best_eval["coordinate_solution"]

            preview_filename = f"preview_{image_filename}"
            preview_path = UPLOAD_DIR / preview_filename
            draw_triangles_preview(image_path, stars, triangles, preview_path)

            top_conf = consensus["candidates"][0]["confidence"] if consensus["candidates"] else 0.0
            top_support = consensus["candidates"][0]["support_count"] if consensus["candidates"] else 0

            # Always display coordinates if available (even if confidence is low)
            # Only set error if truly no solution found
            if coordinate_solution is None:
                error = "Розв'язок не знайдено. Спробуйте інше фото або параметри."

            # Always include coordinate solution in result, with quality indicators
            result = {
                "input_image": url_for("uploaded_file", filename=image_filename),
                "preview_image": url_for("uploaded_file", filename=preview_filename),
                "stars_found": len(stars),
                "triangles_found": len(triangles),
                "consensus": consensus,
                "coordinate_solution": coordinate_solution,  # May contain coords even if low confidence
                "fov_x_deg": fov_x_deg,
                "observation_time_utc": obs_time_utc.isoformat(),
                "time_warning": "Похибка часу 4 хв ≈ 1° по довготі. Перевір UTC-офсет і секунди.",
                "image_filename": image_filename,
                "quality_label": None,
                "auto_tuned": auto_tune,
                "selected_top_n": top_n,
                "selected_tolerance": tolerance,
                "trustworthy": False,
                "confidence": top_conf,  # Include confidence for display
                "support": top_support,  # Include support count for display
            }

            if coordinate_solution is not None:
                fit_rms = float(coordinate_solution.get("fit_rms_deg", 999.0))
                pair_rms = float(coordinate_solution.get("triangle_pair_rms_deg", fit_rms))
                spread = float(coordinate_solution.get("spread_deg", fit_rms))
                hypotheses = int(coordinate_solution.get("used_hypotheses", 1))
                top_conf = consensus["candidates"][0]["confidence"] if consensus["candidates"] else 0.0

                if top_support <= 0:
                    result["quality_label"] = "Низька"
                elif fit_rms < 0.2 and pair_rms < 0.25:
                    result["quality_label"] = "Висока"
                elif fit_rms < 0.7 and pair_rms < 0.7:
                    result["quality_label"] = "Середня"
                else:
                    result["quality_label"] = "Низька"

                result["trustworthy"] = (
                    fit_rms <= 1.2
                    and pair_rms <= 1.0
                    and spread <= 3.0
                    and hypotheses >= 3
                    and top_conf >= 0.35
                )
                result["confidence"] = top_conf
                result["support"] = top_support

            if len(stars) == 0:
                error = "Зірки не знайдені. Спробуйте інше фото або параметри експозиції."
            elif len(triangles) == 0:
                error = "Трикутники не згенеровані. Спробуйте збільшити top_n."

        except Exception as exc:
            error = f"Помилка обробки: {exc}"
        finally:
            pass

    return render_template(
        "index.html",
        result=result,
        error=error,
        form_values=form_values,
        last_image_filename=last_image_filename,
    )


@app.route("/uploads/<path:filename>")
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
