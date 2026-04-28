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
    find_stars_advanced,
    generate_triangles_from_stars,
)
from matcher import StarMatcher
from coordinate_transforms import pixel_to_camera_ray

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp", "webp"}

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = str(UPLOAD_DIR)
app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024
app.secret_key = os.environ.get("STAR_UI_SECRET", "star-ui-dev-secret")


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
    text = re.sub(r"\([^)]*\)", " ", text)
    text = text.replace("LMST", " ").replace("lmst", " ")
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
        return dt.replace(tzinfo=timezone.utc)
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


def evaluate_config(star_db, stars, image_path, fov_x_deg, obs_time_utc, top_n, tolerance):
    probe_img = cv2.imread(str(image_path))
    if probe_img is not None:
        img_h, img_w = probe_img.shape[:2]
        min_sep_px = max(10.0, min(24.0, 0.012 * img_w * (35.0 / max(10.0, fov_x_deg))))
    else:
        min_sep_px = 12.0

    img_shape = probe_img.shape if probe_img is not None else None
    image_rays_by_id = None
    if img_shape is not None:
        image_rays_by_id = {
            star_obj.id: pixel_to_camera_ray(star_obj.center, img_shape, fov_x_deg=fov_x_deg)
            for star_obj in stars
        }

    triangles = generate_triangles_from_stars(stars, top_n=top_n, min_separation_px=min_sep_px)
    triangles_eval = triangles[:220]

    consensus = star_db.find_consensus_matches(
        triangles_eval[: min(12, len(triangles_eval))],
        tolerance=tolerance,
        per_triangle_k=30,
    )

    coordinate_solution = estimate_pointing_multi_triangle(
        star_db=star_db,
        stars=stars,
        image_path=str(image_path),
        triangles=triangles_eval,
        fov_x_deg=fov_x_deg,
        observation_time_utc=obs_time_utc,
        tolerance=tolerance,
        per_triangle_k=12,
        max_fit_rms_deg=6.0,
        img_shape=img_shape,
        image_rays_by_id=image_rays_by_id,
        allow_no_quad=False,
    )

    if coordinate_solution is None:
        coordinate_solution = estimate_pointing_multi_triangle(
            star_db=star_db,
            stars=stars,
            image_path=str(image_path),
            triangles=triangles_eval,
            fov_x_deg=fov_x_deg,
            observation_time_utc=obs_time_utc,
            tolerance=min(0.1, tolerance * 1.8),
            per_triangle_k=30,
            max_fit_rms_deg=12.0,
            img_shape=img_shape,
            image_rays_by_id=image_rays_by_id,
            allow_no_quad=True,
        )

    if coordinate_solution is None:
        for tri in triangles_eval:
            matches = star_db.find_best_match(
                tri["ratio1"],
                tri["ratio2"],
                tri["ratio3"],
                tri["orientation"],
                tolerance=min(0.1, tolerance * 1.6),
                k=5,
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
                img_shape=img_shape,
                image_rays_by_id=image_rays_by_id,
            )
            if coordinate_solution is not None:
                break

    top_conf = consensus["candidates"][0]["confidence"] if consensus["candidates"] else 0.0

    if coordinate_solution is None:
        score = 1e9 - top_conf
    else:
        fit = float(coordinate_solution.get("fit_rms_deg", 99.0))
        pair = float(coordinate_solution.get("triangle_pair_rms_deg", fit))
        quad_support = float(coordinate_solution.get("quad_support_count", 0.0))
        quad_best = float(coordinate_solution.get("quad_support_best_deg", 99.0) or 99.0)
        spread = float(coordinate_solution.get("spread_deg", fit))
        hypotheses = float(coordinate_solution.get("used_hypotheses", 1.0))
        score = (
            fit
            + 0.75 * pair
            + 0.55 * quad_best
            + 0.30 * spread
            + (1.0 - top_conf) * 1.45
            - 0.22 * min(10.0, hypotheses)
            - 0.12 * min(4.0, quad_support)
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
        "demo_mode": session.get("demo_mode", ""),
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
            "demo_mode": "on" if request.form.get("demo_mode") else "",
        }

        session.update(form_values)

        fov_x_deg = parse_float_with_default(form_values["fov_x_deg"], 60.0)
        top_n = max(3, min(30, parse_int_with_default(form_values["top_n"], 8)))
        tolerance = max(0.001, min(0.1, parse_float_with_default(form_values["tolerance"], 0.015)))
        auto_tune = form_values["auto_tune"] == "on"
        demo_mode = form_values["demo_mode"] == "on"
        obs_time_utc = parse_observation_time(form_values["observation_time_utc"])

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

            star_db = StarMatcher(db_config=DB_CONFIG)

            best_eval = evaluate_config(
                star_db=star_db,
                stars=stars,
                image_path=image_path,
                fov_x_deg=fov_x_deg,
                obs_time_utc=obs_time_utc,
                top_n=top_n,
                tolerance=tolerance,
            )

            if auto_tune:
                top_candidates = sorted({max(6, min(24, t)) for t in [top_n - 4, top_n - 2, top_n, top_n + 2, top_n + 4, top_n + 8, top_n + 12]})
                tol_candidates = sorted({max(0.006, min(0.04, v)) for v in [tolerance * 0.60, tolerance * 0.80, tolerance, tolerance * 1.20, tolerance * 1.50, tolerance * 1.80]})

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
                        )
                        if candidate["score"] < best_eval["score"]:
                            best_eval = candidate

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

            result = {
                "input_image": url_for("uploaded_file", filename=image_filename),
                "preview_image": url_for("uploaded_file", filename=preview_filename),
                "stars_found": len(stars),
                "triangles_found": len(triangles),
                "consensus": consensus,
                "coordinate_solution": coordinate_solution,
                "fov_x_deg": fov_x_deg,
                "observation_time_utc": (
                    obs_time_utc.isoformat() if obs_time_utc is not None else "Auto (current UTC)"
                ),
                "image_filename": image_filename,
                "quality_label": None,
                "auto_tuned": auto_tune,
                "selected_top_n": top_n,
                "selected_tolerance": tolerance,
                "trustworthy": False,
                "demo_mode": demo_mode,
                "show_demo_coords": False,
            }

            if coordinate_solution is not None:
                fit_rms = float(coordinate_solution.get("fit_rms_deg", 999.0))
                pair_rms = float(coordinate_solution.get("triangle_pair_rms_deg", fit_rms))
                quad_support = int(coordinate_solution.get("quad_support_count", 0))
                quad_best = float(coordinate_solution.get("quad_support_best_deg", 99.0) or 99.0)
                spread = float(coordinate_solution.get("spread_deg", fit_rms))
                hypotheses = int(coordinate_solution.get("used_hypotheses", 1))
                top_conf = consensus["candidates"][0]["confidence"] if consensus["candidates"] else 0.0

                if fit_rms < 0.2 and pair_rms < 0.25 and quad_best < 0.5:
                    result["quality_label"] = "Висока"
                elif fit_rms < 0.7 and pair_rms < 0.7 and quad_best < 1.0:
                    result["quality_label"] = "Середня"
                else:
                    result["quality_label"] = "Низька"

                result["trustworthy"] = (
                    fit_rms <= 1.2
                    and pair_rms <= 1.0
                    and quad_best <= 1.2
                    and quad_support >= 1
                    and spread <= 3.0
                    and hypotheses >= 3
                    and top_conf >= 0.35
                )

                result["show_demo_coords"] = result["trustworthy"] or demo_mode

            if len(stars) == 0:
                error = "Зірки не знайдені. Спробуйте інше фото або параметри експозиції."
            elif len(triangles) == 0:
                error = "Трикутники не згенеровані. Спробуйте збільшити top_n."
            elif not consensus["candidates"]:
                error = "Збіги в каталозі не знайдено для цього кадру."

        except Exception as exc:
            error = f"Помилка обробки: {exc}"
        finally:
            if star_db is not None:
                star_db.close()

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
