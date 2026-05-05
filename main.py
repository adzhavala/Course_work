import cv2
import numpy as np
import math
from datetime import datetime, timezone
from itertools import combinations
from itertools import permutations
from collections import Counter, defaultdict
from coordinate_transforms import (
    cartesian_to_ra_dec,
    ecef_to_lat_lon_deg,
    eci_to_ecef,
    normalize_vector,
    pixel_to_camera_ray,
    solve_vector_from_angular_constraints,
)

PAIR_RMS_GATE_DEG = 3.0
RANK_MISMATCH_WEIGHT = 0.05
FOV_DIAG_MARGIN = 1.1


class Star:
    def __init__(self, star_id, center, radius, brightness):
        self.id = star_id
        self.center = center
        self.radius = radius
        self.brightness = brightness

    def __repr__(self):
        return f"Star(ID={self.id}, Center={self.center}, R={self.radius:.2f}, Bright={self.brightness:.2f})"


def find_stars_advanced(
    path,
    threshold_factor=4.0,
    min_area=8,
    max_area=10000,
    border_exclusion_frac=0.02,
    min_circularity=0.5,
    max_axis_ratio=5.0,
):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Failed to open image")

    # Auto-suppress UI overlays (top/bottom status bars) using row brightness.
    h, w = img.shape[:2]
    row_means = img.mean(axis=1)
    median = float(np.median(row_means))
    std = float(np.std(row_means))
    threshold = median + 2.0 * std
    max_crop = int(h * 0.15)

    top_cut = 0
    for i in range(max_crop):
        if row_means[i] > threshold:
            top_cut = i + 1
        elif top_cut > 0 and i - top_cut > 3:
            break

    bottom_cut = 0
    for i in range(h - 1, max(h - max_crop, 0), -1):
        if row_means[i] > threshold:
            bottom_cut = h - i
        elif bottom_cut > 0 and h - i - bottom_cut > 3:
            break

    if top_cut > 0:
        img[:top_cut, :] = 0
    if bottom_cut > 0:
        img[h - bottom_cut:, :] = 0

    blurred = cv2.GaussianBlur(img, (3, 3), 0)
    mean, std = cv2.meanStdDev(blurred)

    mean_val = mean[0][0]
    std_val = std[0][0]
    T = mean_val + threshold_factor * std_val

    _, binary = cv2.threshold(blurred, T, 255, cv2.THRESH_BINARY)

    # Ignore border zones where Stellarium/OS overlays are commonly rendered.
    h, w = binary.shape[:2]
    by = int(h * border_exclusion_frac)
    bx = int(w * border_exclusion_frac)
    if by > 0:
        binary[:by, :] = 0
        binary[h - by:, :] = 0
    if bx > 0:
        binary[:, :bx] = 0
        binary[:, w - bx:] = 0

    kernel = np.ones((2, 2), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary)

    stars_data = []

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]

        if area < min_area or area > max_area:
            continue

        comp_w = stats[i, cv2.CC_STAT_WIDTH]
        comp_h = stats[i, cv2.CC_STAT_HEIGHT]
        axis_ratio = max(comp_w, comp_h) / max(1, min(comp_w, comp_h))
        if axis_ratio > max_axis_ratio:
            # Text/UI strokes are usually elongated, true star blobs are compact.
            continue

        cx = float(centroids[i][0])
        cy = float(centroids[i][1])

        star_mask = (labels == i).astype(np.uint8) * 255

        contours, _ = cv2.findContours(star_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        perimeter = cv2.arcLength(contours[0], True)
        if perimeter <= 1e-9:
            continue
        circularity = 4.0 * math.pi * area / (perimeter * perimeter)
        if circularity < min_circularity:
            continue

        points = cv2.findNonZero(star_mask)

        if points is not None and len(points) > 0:
            points = points.reshape(-1, 2).astype(np.float64)
            distances = np.linalg.norm(points - np.array([cx, cy]), axis=1)
            max_radius = float(np.max(distances))
        else:
            max_radius = 0.0

        _, max_val, _, _ = cv2.minMaxLoc(img, mask=star_mask)

        raw_brightness = float(max_val)
        raw_flux = raw_brightness * float(area)

        stars_data.append({
            "id": i,
            "center": (cx, cy),
            "radius": max_radius,
            "area": int(area),
            "raw_brightness": raw_brightness,
            "raw_flux": raw_flux,
        })

    if not stars_data:
        return []

    stars_data.sort(key=lambda s: s["raw_flux"], reverse=True)
    all_flux = [s["raw_flux"] for s in stars_data]
    min_b = min(all_flux)
    max_b = max(all_flux)

    final_stars = []
    for s in stars_data:
        if max_b - min_b > 0:
            norm_b = (s["raw_flux"] - min_b) / (max_b - min_b)
        else:
            norm_b = 1.0

        new_star = Star(s["id"], s["center"], s["radius"], norm_b)
        final_stars.append(new_star)

    return final_stars

def calculate_distance(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def select_spread_bright_stars(stars, top_n=10, min_separation_px=16.0):
    """Greedy bright-star selection with minimum spacing to reduce near-duplicate picks."""
    sorted_by_brightness = sorted(stars, key=lambda s: s.brightness, reverse=True)
    selected = []

    for star in sorted_by_brightness:
        too_close = any(calculate_distance(star.center, s.center) < min_separation_px for s in selected)
        if too_close:
            continue
        selected.append(star)
        if len(selected) >= top_n:
            break

    # Fallback: if scene is dense and spacing rejected too much, fill by brightness.
    if len(selected) < max(3, top_n):
        used_ids = {s.id for s in selected}
        for star in sorted_by_brightness:
            if star.id in used_ids:
                continue
            selected.append(star)
            if len(selected) >= top_n:
                break

    return selected


def alpha_from_pixel(center_xy, image_shape, fov_x_deg=60.0):
    """Angle between optical axis and star ray from pixel position."""
    height, width = image_shape[:2]
    cx = width / 2.0
    cy = height / 2.0

    f = (width / 2.0) / math.tan(math.radians(fov_x_deg / 2.0))
    dx = center_xy[0] - cx
    dy = center_xy[1] - cy
    radial_px = math.sqrt(dx * dx + dy * dy)

    return math.degrees(math.atan2(radial_px, f))


def fov_diag_limit_deg(fov_x_deg, image_shape, margin=FOV_DIAG_MARGIN):
    """Compute a diagonal FOV limit for triangle filtering from image shape."""
    if image_shape is None:
        return fov_x_deg * 1.6

    height, width = image_shape[:2]
    if width <= 0 or height <= 0 or fov_x_deg <= 0.0 or fov_x_deg >= 179.0:
        return fov_x_deg * 1.6

    tan_x = math.tan(math.radians(fov_x_deg / 2.0))
    tan_y = tan_x * (height / width)
    tan_diag = math.hypot(tan_x, tan_y)
    fov_diag_deg = math.degrees(2.0 * math.atan(tan_diag))
    return fov_diag_deg * margin


def estimate_pointing_and_earth_coordinates(
    star_db,
    stars,
    image_path,
    image_triangle,
    hip_triangle,
    fov_x_deg=60.0,
    observation_time_utc=None,
    force=False,
    pair_rms_gate_deg=PAIR_RMS_GATE_DEG,
):
    """Estimate optical axis in ECI via intersection of position circles.

    The solver uses angular distances from the optical axis (image center) to each
    matched star. Mapping the resulting optical axis to Earth yields the observer's
    location ONLY if the camera was pointing at the local zenith.
    """
    img = cv2.imread(image_path)
    if img is None:
        return None

    star_by_id = {s.id: s for s in stars}
    if len(image_triangle) != 3 or len(hip_triangle) != 3:
        return None

    image_star_ids = list(image_triangle)
    alphas_deg = []
    image_brightness = []
    image_rays = []
    for img_star_id in image_star_ids:
        star_obj = star_by_id.get(img_star_id)
        if star_obj is None:
            return None
        alphas_deg.append(alpha_from_pixel(star_obj.center, img.shape, fov_x_deg=fov_x_deg))
        image_brightness.append(star_obj.brightness)
        image_rays.append(pixel_to_camera_ray(star_obj.center, img.shape, fov_x_deg=fov_x_deg))

    hip_ids = [int(h) for h in hip_triangle]
    vectors_map = star_db.get_star_vectors_by_hips(hip_ids)
    mags_map = star_db.get_star_magnitudes_by_hips(hip_ids)
    if len(vectors_map) != len(hip_ids):
        return None

    # Strict angular consistency gate: reject catalog triangles with mismatched side angles.
    base_star_vectors = [vectors_map[hip] for hip in hip_ids]
    pair_rms_deg = _triangle_pair_rms_deg(image_rays, base_star_vectors)
    if pair_rms_deg > pair_rms_gate_deg and not force:
        return None

    # Triangle features are permutation-invariant, so resolve correspondence explicitly.
    best = None
    # Brightness order: image brightness is higher for brighter stars.
    img_rank = sorted(range(len(image_brightness)), key=lambda i: -image_brightness[i])

    for perm in permutations(hip_ids):
        star_vectors = [vectors_map[hip] for hip in perm]

        if len(mags_map) == len(hip_ids):
            mags = [mags_map[hip] for hip in perm]
            cat_rank = sorted(range(len(mags)), key=lambda i: mags[i])
            rank_mismatch = sum(1 for a, b in zip(img_rank, cat_rank) if a != b)
        else:
            rank_mismatch = 0

        try:
            r_eci, residuals = solve_vector_from_angular_constraints(star_vectors, alphas_deg, radius=1.0)
        except Exception:
            continue

        residuals = np.asarray(residuals, dtype=np.float64)
        residual_rms = float(np.sqrt(np.mean(np.square(residuals))))
        residual_max = float(np.max(np.abs(residuals)))

        # Angular residuals for reporting (degrees).
        ang_residuals = []
        for vec, alpha_obs in zip(star_vectors, alphas_deg):
            cos_alpha = float(np.dot(normalize_vector(vec), normalize_vector(r_eci)))
            cos_alpha = max(-1.0, min(1.0, cos_alpha))
            alpha_hat = math.degrees(math.acos(cos_alpha))
            ang_residuals.append(alpha_hat - alpha_obs)
        ang_residuals = np.asarray(ang_residuals, dtype=np.float64)
        angle_rms_deg = float(np.sqrt(np.mean(np.square(ang_residuals))))
        angle_max_deg = float(np.max(np.abs(ang_residuals)))

        # Composite score: angular fit + pairwise consistency + weak brightness tie-breaker.
        score = angle_rms_deg + 0.5 * pair_rms_deg + RANK_MISMATCH_WEIGHT * rank_mismatch

        if best is None or score < best['score']:
            best = {
                'perm': perm,
                'r_eci': r_eci,
                'residual_rms': residual_rms,
                'residual_max': residual_max,
                'angle_rms_deg': angle_rms_deg,
                'angle_max_deg': angle_max_deg,
                'score': score,
            }

    if best is None:
        return None

    r_eci = normalize_vector(best['r_eci'], radius=1.0)

    ra_deg, dec_deg = cartesian_to_ra_dec(r_eci)

    now_utc = observation_time_utc or datetime.now(timezone.utc)
    if now_utc.tzinfo is None:
        now_utc = now_utc.replace(tzinfo=timezone.utc)
    else:
        now_utc = now_utc.astimezone(timezone.utc)
    r_ecef = eci_to_ecef(r_eci, now_utc)
    pointing_lat_deg, pointing_lon_deg = ecef_to_lat_lon_deg(r_ecef)

    return {
        'ra_deg': ra_deg,
        'dec_deg': dec_deg,
        'lat_deg': pointing_lat_deg,
        'lon_deg': pointing_lon_deg,
        'pointing_lat_deg': pointing_lat_deg,
        'pointing_lon_deg': pointing_lon_deg,
        'r_eci': r_eci,
        'fit_rms_deg': best['angle_rms_deg'],
        'fit_max_deg': best['angle_max_deg'],
        'residual_rms': best['residual_rms'],
        'residual_max': best['residual_max'],
        'triangle_pair_rms_deg': pair_rms_deg,
        'timestamp_utc': now_utc.isoformat(),
        'used_hips': tuple(best['perm']),
        'matched_hips': tuple(best['perm']),
        'image_star_ids': tuple(image_star_ids),
    }


def _vector_angle_deg(v1, v2):
    a = np.asarray(v1, dtype=np.float64)
    b = np.asarray(v2, dtype=np.float64)
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    c = float(np.dot(a, b))
    c = max(-1.0, min(1.0, c))
    return math.degrees(math.acos(c))


def _triangle_pair_angles_deg(vectors):
    """Return the 3 pairwise angular distances for a 3-vector set."""
    if len(vectors) != 3:
        raise ValueError("Expected exactly 3 vectors")

    v0, v1, v2 = vectors
    return np.array([
        _vector_angle_deg(v0, v1),
        _vector_angle_deg(v0, v2),
        _vector_angle_deg(v1, v2),
    ], dtype=np.float64)


def _triangle_pair_rms_deg(vectors_a, vectors_b):
    """RMS mismatch between pairwise angular distances of two 3-point sets.

    Order-invariant: sorts the 3 angles to avoid dependence on vertex ordering.
    """
    a = np.sort(_triangle_pair_angles_deg(vectors_a))
    b = np.sort(_triangle_pair_angles_deg(vectors_b))
    diff = a - b
    return float(np.sqrt(np.mean(np.square(diff))))


def estimate_pointing_multi_triangle(
    star_db,
    stars,
    image_path,
    triangles,
    fov_x_deg=60.0,
    observation_time_utc=None,
    tolerance=0.015,
    per_triangle_k=3,
    max_fit_rms_deg=5.0,
    pair_rms_gate_deg=PAIR_RMS_GATE_DEG,
    max_side_deg=None,
):
    """Aggregate multiple triangle hypotheses into one robust coordinate estimate."""
    hypotheses = []

    max_side_limit_deg = max_side_deg
    if max_side_limit_deg is None:
        probe_img = cv2.imread(image_path)
        if probe_img is not None:
            max_side_limit_deg = fov_diag_limit_deg(fov_x_deg, probe_img.shape)

    for t in triangles:
        # Call find_best_match with 3D ratio features and FOV for max_side_deg filtering.
        # Orientation is no longer passed; the KD-Tree operates in pure ratio space.
        matches = star_db.find_best_match(
            t['ratio1'],
            t['ratio2'],
            t['ratio3'],
            tolerance=tolerance,
            fov_x_deg=fov_x_deg,
            max_side_deg=max_side_limit_deg,
            k=per_triangle_k,
        )
        if not matches:
            continue

        for m in matches:
            sol = estimate_pointing_and_earth_coordinates(
                star_db=star_db,
                stars=stars,
                image_path=image_path,
                image_triangle=t['star_ids'],
                hip_triangle=m['star_hips'],
                fov_x_deg=fov_x_deg,
                observation_time_utc=observation_time_utc,
                pair_rms_gate_deg=pair_rms_gate_deg,
            )
            if sol is None:
                continue
            if sol['fit_rms_deg'] > max_fit_rms_deg:
                continue
            if float(sol.get('triangle_pair_rms_deg', sol['fit_rms_deg'])) > pair_rms_gate_deg:
                continue

            hypotheses.append({
                'solution': sol,
                'match_error': float(m['error']),
                'triangle_id': t['triangle_id'],
            })

    if len(hypotheses) < 2:
        return None

    hips_support = Counter(tuple(sorted(h['solution']['used_hips'])) for h in hypotheses)

    vectors = []
    weights = []
    for h in hypotheses:
        fit = h['solution']['fit_rms_deg']
        pair = float(h['solution'].get('triangle_pair_rms_deg', fit))
        err = h['match_error']
        hips_key = tuple(sorted(h['solution']['used_hips']))
        support_boost = 1.0 + 0.35 * max(0, hips_support[hips_key] - 1)
        weight = support_boost / ((0.2 + fit) ** 2 * (0.2 + pair) * (0.005 + err))
        vectors.append(np.asarray(h['solution']['r_eci'], dtype=np.float64))
        weights.append(weight)

    vectors = np.asarray(vectors, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)

    # Keep only the densest angular cluster to suppress ambiguous false matches.
    inlier_threshold_deg = 6.0
    best_inliers = None
    best_score = -1.0

    for i in range(len(vectors)):
        angles = np.array([_vector_angle_deg(vectors[i], vj) for vj in vectors], dtype=np.float64)
        inliers = angles <= inlier_threshold_deg
        score = float(np.sum(weights[inliers]))
        if score > best_score:
            best_score = score
            best_inliers = inliers

    if best_inliers is None or int(np.sum(best_inliers)) == 0:
        best_inliers = np.ones(len(vectors), dtype=bool)

    vectors_in = vectors[best_inliers]
    weights_in = weights[best_inliers]

    weighted_sum = np.zeros(3, dtype=np.float64)
    for v, wv in zip(vectors_in, weights_in):
        weighted_sum += wv * v

    if np.linalg.norm(weighted_sum) < 1e-12:
        return None

    r_eci_agg = weighted_sum / np.linalg.norm(weighted_sum)

    now_utc = observation_time_utc or datetime.now(timezone.utc)
    if now_utc.tzinfo is None:
        now_utc = now_utc.replace(tzinfo=timezone.utc)
    else:
        now_utc = now_utc.astimezone(timezone.utc)

    hypotheses_in = [h for h, ok in zip(hypotheses, best_inliers) if ok]
    fit_values = [h['solution']['fit_rms_deg'] for h in hypotheses_in]
    pair_values = [float(h['solution'].get('triangle_pair_rms_deg', h['solution']['fit_rms_deg'])) for h in hypotheses_in]
    best_h = min(hypotheses_in, key=lambda x: x['solution']['fit_rms_deg'])

    r_eci_final = r_eci_agg
    fit_rms_deg = float(np.mean(fit_values)) if fit_values else 99.0
    fit_max_deg = float(np.max(fit_values)) if fit_values else 99.0
    residual_rms = None
    residual_max = None

    # Use all consensus star correspondences (N>=4) to over-constrain the algebraic solve.
    img = cv2.imread(image_path)
    if img is not None:
        star_by_id = {s.id: s for s in stars}
        match_votes = defaultdict(Counter)

        for h in hypotheses_in:
            img_ids = h['solution'].get('image_star_ids')
            hip_ids = h['solution'].get('matched_hips')
            if not img_ids or not hip_ids:
                continue
            for img_id, hip_id in zip(img_ids, hip_ids):
                match_votes[img_id][hip_id] += 1

        candidate_pairs = []
        for img_id, counter in match_votes.items():
            hip_id, votes = counter.most_common(1)[0]
            candidate_pairs.append((votes, img_id, hip_id))
        candidate_pairs.sort(reverse=True)

        pairs = []
        used_hips = set()
        for votes, img_id, hip_id in candidate_pairs:
            if hip_id in used_hips:
                continue
            used_hips.add(hip_id)
            pairs.append((votes, img_id, hip_id))

        strong_pairs = [p for p in pairs if p[0] >= 2]
        if len(strong_pairs) >= 4:
            pairs = strong_pairs

        if len(pairs) >= 4:
            hip_ids = [hip_id for _, _, hip_id in pairs]
            vectors_map = star_db.get_star_vectors_by_hips(hip_ids)
            star_vectors = []
            alphas_deg = []

            for _, img_id, hip_id in pairs:
                star_obj = star_by_id.get(img_id)
                vec = vectors_map.get(hip_id)
                if star_obj is None or vec is None:
                    continue
                alphas_deg.append(alpha_from_pixel(star_obj.center, img.shape, fov_x_deg=fov_x_deg))
                star_vectors.append(vec)

            if len(star_vectors) >= 4:
                try:
                    r_ref, residuals = solve_vector_from_angular_constraints(star_vectors, alphas_deg, radius=1.0)
                    r_ref = normalize_vector(r_ref, radius=1.0)

                    residuals = np.asarray(residuals, dtype=np.float64)
                    residual_rms = float(np.sqrt(np.mean(np.square(residuals))))
                    residual_max = float(np.max(np.abs(residuals)))

                    ang_residuals = []
                    for vec, alpha_obs in zip(star_vectors, alphas_deg):
                        cos_alpha = float(np.dot(normalize_vector(vec), r_ref))
                        cos_alpha = max(-1.0, min(1.0, cos_alpha))
                        alpha_hat = math.degrees(math.acos(cos_alpha))
                        ang_residuals.append(alpha_hat - alpha_obs)
                    ang_residuals = np.asarray(ang_residuals, dtype=np.float64)
                    fit_rms_deg = float(np.sqrt(np.mean(np.square(ang_residuals))))
                    fit_max_deg = float(np.max(np.abs(ang_residuals)))

                    r_eci_final = r_ref
                except Exception:
                    pass

    ra_deg, dec_deg = cartesian_to_ra_dec(r_eci_final)
    r_ecef = eci_to_ecef(r_eci_final, now_utc)
    pointing_lat_deg, pointing_lon_deg = ecef_to_lat_lon_deg(r_ecef)

    spread_values = [_vector_angle_deg(h['solution']['r_eci'], r_eci_final) for h in hypotheses_in]

    return {
        'ra_deg': ra_deg,
        'dec_deg': dec_deg,
        'lat_deg': pointing_lat_deg,
        'lon_deg': pointing_lon_deg,
        'pointing_lat_deg': pointing_lat_deg,
        'pointing_lon_deg': pointing_lon_deg,
        'r_eci': r_eci_final,
        'fit_rms_deg': fit_rms_deg,
        'fit_max_deg': fit_max_deg,
        'residual_rms': residual_rms,
        'residual_max': residual_max,
        'triangle_pair_rms_deg': float(np.mean(pair_values)) if pair_values else fit_rms_deg,
        'spread_deg': float(np.mean(spread_values)) if spread_values else 0.0,
        'timestamp_utc': now_utc.isoformat(),
        'used_hips': best_h['solution']['used_hips'],
        'used_hypotheses': len(hypotheses_in),
        'total_hypotheses': len(hypotheses),
        'max_triplet_support': int(max(hips_support.values()) if hips_support else 1),
        'best_triangle_id': best_h['triangle_id'],
    }


def generate_triangles_from_stars(
    stars,
    image_shape,
    fov_x_deg,
    top_n=10,
    min_separation_px=12.0,
):
    """Generate triangles from image stars using 3D angular distances.
    
    Orientation is no longer computed for image triangles, since the KD-Tree
    now operates in pure 3D ratio space, avoiding incompatibilities between
    spherical and pixel coordinate conventions.
    """
    brightest_stars = select_spread_bright_stars(stars, top_n=top_n, min_separation_px=min_separation_px)

    triangles_data = []

    for star_combo in combinations(brightest_stars, 3):
        s1, s2, s3 = star_combo

        r1 = pixel_to_camera_ray(s1.center, image_shape, fov_x_deg=fov_x_deg)
        r2 = pixel_to_camera_ray(s2.center, image_shape, fov_x_deg=fov_x_deg)
        r3 = pixel_to_camera_ray(s3.center, image_shape, fov_x_deg=fov_x_deg)

        d1 = _vector_angle_deg(r1, r2)
        d2 = _vector_angle_deg(r2, r3)
        d3 = _vector_angle_deg(r1, r3)

        sides = sorted([d1, d2, d3])
        a, b, c = sides[0], sides[1], sides[2]

        if a < 0.001:
            continue

        cos_gamma = (a**2 + b**2 - c**2) / (2 * a * b)
        cos_gamma = max(-1.0, min(1.0, cos_gamma))
        max_angle = math.degrees(math.acos(cos_gamma))

        cos_alpha = (b**2 + c**2 - a**2) / (2 * b * c)
        cos_alpha = max(-1.0, min(1.0, cos_alpha))
        min_angle = math.degrees(math.acos(cos_alpha))

        if min_angle < 10.0 or max_angle > 170.0:
            continue

        ratio1 = b / a
        ratio2 = c / a
        ratio3 = b / c if c > 0 else 0.0
        # Orientation is NO LONGER COMPUTED for image triangles

        triangle_id = f"tri_{s1.id}_{s2.id}_{s3.id}"

        triangles_data.append({
            'triangle_id': triangle_id,
            'star_ids': (s1.id, s2.id, s3.id),
            'ratio1': ratio1,
            'ratio2': ratio2,
            'ratio3': ratio3,
            # No 'orientation' field
        })

    return triangles_data


if __name__ == "__main__":
    from matcher import StarMatcher
    from config_db import DB_CONFIG

    try:
        star_db = StarMatcher(db_config=DB_CONFIG)
    except Exception as e:
        print(f"Matcher initialization error: {e}")
        exit(1)

    try:
        if not __import__('os').path.exists("stars.png"):
            print("File 'stars.png' not found")
            star_db.close()
            exit(1)

        stars = find_stars_advanced("stars.png")
        print(f"Total stars found: {len(stars)}")

        if not stars:
            print("Stars not found in image. Check filter parameters.")
            star_db.close()
            exit(1)

        img_shape = cv2.imread("stars.png").shape
        max_side_deg = fov_diag_limit_deg(60.0, img_shape)
        triangles = generate_triangles_from_stars(stars, img_shape, fov_x_deg=60.0, top_n=8)

        if not triangles:
            print("Triangles not generated. Too few bright stars?")
            star_db.close()
            exit(1)

        print(f"Generated valid triangles: {len(triangles)}")
        print("First 3 triangles (their ratios):")

        matches_found = 0
        for i in range(min(3, len(triangles))):
            t = triangles[i]
            print(f"\n  Triangle {i+1}: ratio1={t['ratio1']:.3f}, ratio2={t['ratio2']:.3f}")
            # Call with 3D ratios and FOV; orientation is no longer used
            matches = star_db.find_best_match(
                t['ratio1'],
                t['ratio2'],
                t['ratio3'],
                tolerance=0.015,
                fov_x_deg=60.0,
                max_side_deg=max_side_deg,
                k=20
            )
            if matches:
                matches_found += 1
                print(f"Found {len(matches)} matches!")
            else:
                print(f"No matches found")

        print(f"\nResult: {matches_found}/3 triangles found matches in DB")

        consensus_triangles = triangles[:min(12, len(triangles))]
        consensus = star_db.find_consensus_matches(
            consensus_triangles,
            tolerance=0.015,
            per_triangle_k=20,
            fov_x_deg=60.0,
            max_side_deg=max_side_deg,
        )

        print("\nFinal confidence from multiple triangles:")
        print(
            f"  Triangles used: {consensus['used_triangles']}, "
            f"matched: {consensus['matched_triangles']}"
        )

        if consensus['candidates']:
            top_candidates = consensus['candidates'][:3]
            for i, candidate in enumerate(top_candidates, 1):
                print(
                    f"  {i}) HIP: {candidate['star_hips']} | "
                    f"support={candidate['support_count']}/{consensus['used_triangles']} | "
                    f"avg_error={candidate['avg_error']:.6f} | "
                    f"confidence={candidate['confidence']:.3f}"
                )

            # Robust integration: aggregate several triangle hypotheses.
            try:
                observation_time_utc = datetime.now(timezone.utc)

                best_solution = estimate_pointing_multi_triangle(
                    star_db=star_db,
                    stars=stars,
                    image_path="stars.png",
                    triangles=triangles,
                    fov_x_deg=60.0,
                    observation_time_utc=observation_time_utc,
                    tolerance=0.015,
                    per_triangle_k=3,
                    max_fit_rms_deg=6.0,
                    max_side_deg=max_side_deg,
                )

                if best_solution is not None:
                    print("\nCoordinate-transform solution (improved fit):")
                    print(
                        f"  UTC: {best_solution['timestamp_utc']} | "
                        f"RA={best_solution['ra_deg']:.4f}°, Dec={best_solution['dec_deg']:.4f}°"
                    )
                    print(
                        f"  Observer location: lat={best_solution['lat_deg']:.4f}°, "
                        f"lon={best_solution['lon_deg']:.4f}°"
                    )
                    if best_solution.get('zenith_ra_deg') is not None:
                        print(
                            f"  Zenith RA/Dec: RA={best_solution['zenith_ra_deg']:.4f}°, "
                            f"Dec={best_solution['zenith_dec_deg']:.4f}°"
                        )
                    print(
                        f"  fit_rms={best_solution['fit_rms_deg']:.4f}°, "
                        f"fit_max={best_solution['fit_max_deg']:.4f}°, "
                        f"spread={best_solution['spread_deg']:.4f}°, "
                        f"hypotheses={best_solution['used_hypotheses']}, "
                        f"triangle={best_solution['best_triangle_id']}, "
                        f"HIPs={best_solution['used_hips']}"
                    )
                else:
                    print("\nCoordinate-transform step skipped: no anchor triangle match.")
            except Exception as e:
                print(f"\nCoordinate-transform step failed: {e}")
        else:
            print("  No final candidates found")

    except FileNotFoundError as e:
        print(f"File not found: {e}")
        star_db.close()
        exit(1)
    except Exception as e:
        import traceback
        print(f"Critical error: {e}")
        traceback.print_exc()
        star_db.close()
        exit(1)

    finally:
        try:
            if len(triangles) > 0:
                img_check = cv2.imread("stars.png")
                if img_check is None:
                    print("Failed to read image for visualization")
                else:
                    colors = [(0, 255, 0), (255, 100, 0), (255, 0, 255)]
                    num_triangles_to_draw = min(3, len(triangles))

                    for i in range(num_triangles_to_draw):
                        triangle_ids = triangles[i]['star_ids']
                        color = colors[i]

                        points = []
                        for s in stars:
                            if s.id in triangle_ids:
                                points.append((int(s.center[0]), int(s.center[1])))

                        if len(points) == 3:
                            p1, p2, p3 = points[0], points[1], points[2]
                            cv2.line(img_check, p1, p2, color, 1)
                            cv2.line(img_check, p2, p3, color, 1)
                            cv2.line(img_check, p3, p1, color, 1)
                            cv2.circle(img_check, p1, 3, color, -1)
                            cv2.circle(img_check, p2, 3, color, -1)
                            cv2.circle(img_check, p3, 3, color, -1)

                    cv2.imwrite("triangle_check.png", img_check)
                    print(f"Success! Drew {num_triangles_to_draw} triangles. Open 'triangle_check.png'.")
        except Exception as e:
            print(f"Visualization error: {e}")
        finally:
            star_db.close()
            print("Program completed.")
