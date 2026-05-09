import cv2
import numpy as np
import math
from datetime import datetime, timezone
from itertools import combinations
from itertools import permutations
from collections import Counter
from coordinate_transforms import (
    angular_residuals_deg,
    cartesian_to_ra_dec,
    ecef_to_lat_lon_deg,
    estimate_rotation_kabsch,
    eci_to_ecef,
    pixel_to_camera_ray,
)


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
    threshold_factor=2.5,
    min_area=2,
    max_area=500,
    border_exclusion_frac=0.02,
    min_circularity=0.10,
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

        stars_data.append({
            "id": i,
            "center": (cx, cy),
            "radius": max_radius,
            "raw_brightness": float(max_val)
        })

    if not stars_data:
        return []

    all_brights = [s["raw_brightness"] for s in stars_data]
    min_b = min(all_brights)
    max_b = max(all_brights)

    final_stars = []
    for s in stars_data:
        if max_b - min_b > 0:
            norm_b = (s["raw_brightness"] - min_b) / (max_b - min_b)
        else:
            norm_b = 1.0

        new_star = Star(s["id"], s["center"], s["radius"], norm_b)
        final_stars.append(new_star)

    return final_stars

def calculate_distance(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def orientation_sign_2d(p1, p2, p3):
    cross = (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0])
    if cross == 0:
        return 0.0
    return 1.0 if cross > 0 else -1.0


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


def estimate_pointing_and_earth_coordinates(
    star_db,
    stars,
    image_path,
    image_triangle,
    hip_triangle,
    fov_x_deg=60.0,
    observation_time_utc=None,
):
    """Estimate camera pointing in celestial frame and map it to Earth coordinates."""
    img = cv2.imread(image_path)
    if img is None:
        return None

    star_by_id = {s.id: s for s in stars}
    if len(image_triangle) != 3 or len(hip_triangle) != 3:
        return None

    image_star_ids = list(image_triangle)
    image_rays = []
    for img_star_id in image_star_ids:
        star_obj = star_by_id.get(img_star_id)
        if star_obj is None:
            return None
        image_rays.append(pixel_to_camera_ray(star_obj.center, img.shape, fov_x_deg=fov_x_deg))

    image_brightness = [star_by_id[sid].brightness for sid in image_star_ids]
    observed_pair_angles = _triangle_pair_angles_deg(image_rays)

    hip_ids = [int(h) for h in hip_triangle]
    vectors_map = star_db.get_star_vectors_by_hips(hip_ids)
    mags_map = star_db.get_star_magnitudes_by_hips(hip_ids)
    if len(vectors_map) != len(hip_ids):
        return None

    # Triangle features are permutation-invariant, so resolve correspondence explicitly.
    best = None
    # Brightness order: image brightness is higher for brighter stars.
    img_rank = sorted(range(len(image_brightness)), key=lambda i: -image_brightness[i])

    for perm in permutations(hip_ids):
        star_vectors = [vectors_map[hip] for hip in perm]
        pair_rms_deg = _triangle_pair_rms_deg(image_rays, star_vectors)
        if pair_rms_deg > 100.0: # Replaced 1.6 to allow all
            # Reject catalog triangles that do not preserve observed angular scale.
            continue

        if len(mags_map) == len(hip_ids):
            mags = [mags_map[hip] for hip in perm]
            cat_rank = sorted(range(len(mags)), key=lambda i: mags[i])
            rank_mismatch = sum(1 for a, b in zip(img_rank, cat_rank) if a != b)
        else:
            rank_mismatch = 0

        try:
            r_ci = estimate_rotation_kabsch(star_vectors, image_rays)
            residuals_deg = angular_residuals_deg(star_vectors, image_rays, r_ci)
            rms_deg = float(np.sqrt(np.mean(np.square(residuals_deg))))
        except Exception:
            continue

        # Composite score: fit + pair-angle consistency + brightness-order consistency.
        score = rms_deg + 0.8 * pair_rms_deg + 0.35 * rank_mismatch

        if best is None or score < best['score']:
            best = {
                'perm': perm,
                'rotation': r_ci,
                'residuals_deg': residuals_deg,
                'rms_deg': rms_deg,
                'pair_rms_deg': pair_rms_deg,
                'score': score,
            }

    if best is None:
        return None

    r_ci = best['rotation']
    # Optical axis in camera frame is +Z, convert to inertial using inverse rotation.
    optical_axis_camera = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    r_eci = r_ci.T @ optical_axis_camera
    r_eci = r_eci / np.linalg.norm(r_eci)

    ra_deg, dec_deg = cartesian_to_ra_dec(r_eci)

    now_utc = observation_time_utc or datetime.now(timezone.utc)
    if now_utc.tzinfo is None:
        now_utc = now_utc.replace(tzinfo=timezone.utc)
    else:
        now_utc = now_utc.astimezone(timezone.utc)
    r_ecef = eci_to_ecef(r_eci, now_utc)
    lat_deg, lon_deg = ecef_to_lat_lon_deg(r_ecef)

    return {
        'ra_deg': ra_deg,
        'dec_deg': dec_deg,
        'lat_deg': lat_deg,
        'lon_deg': lon_deg,
        'r_eci': r_eci,
        'fit_rms_deg': best['rms_deg'],
        'fit_max_deg': float(np.max(best['residuals_deg'])),
        'triangle_pair_rms_deg': best['pair_rms_deg'],
        'timestamp_utc': now_utc.isoformat(),
        'used_hips': tuple(best['perm']),
        'observed_pair_angles_deg': observed_pair_angles.tolist(),
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
    """RMS mismatch between pairwise angular distances of two 3-point sets."""
    a = _triangle_pair_angles_deg(vectors_a)
    b = _triangle_pair_angles_deg(vectors_b)
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
):
    """Aggregate multiple triangle hypotheses into one robust coordinate estimate."""
    hypotheses = []

    for t in triangles:
        matches = star_db.find_best_match(
            t['ratio1'],
            t['ratio2'],
            t['ratio3'],
            tolerance=tolerance,
            fov_x_deg=fov_x_deg,
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
            )
            if sol is None:
                continue
            if sol['fit_rms_deg'] > max_fit_rms_deg:
                pass # continue
            if float(sol.get('triangle_pair_rms_deg', sol['fit_rms_deg'])) > 1.2:
                pass # continue

            hypotheses.append({
                'solution': sol,
                'match_error': float(m['error']),
                'triangle_id': t['triangle_id'],
            })

    if not hypotheses:
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
    ra_deg, dec_deg = cartesian_to_ra_dec(r_eci_agg)

    now_utc = observation_time_utc or datetime.now(timezone.utc)
    if now_utc.tzinfo is None:
        now_utc = now_utc.replace(tzinfo=timezone.utc)
    else:
        now_utc = now_utc.astimezone(timezone.utc)

    r_ecef = eci_to_ecef(r_eci_agg, now_utc)
    lat_deg, lon_deg = ecef_to_lat_lon_deg(r_ecef)

    hypotheses_in = [h for h, ok in zip(hypotheses, best_inliers) if ok]
    fit_values = [h['solution']['fit_rms_deg'] for h in hypotheses_in]
    pair_values = [float(h['solution'].get('triangle_pair_rms_deg', h['solution']['fit_rms_deg'])) for h in hypotheses_in]
    spread_values = [_vector_angle_deg(h['solution']['r_eci'], r_eci_agg) for h in hypotheses_in]
    best_h = min(hypotheses_in, key=lambda x: x['solution']['fit_rms_deg'])

    return {
        'ra_deg': ra_deg,
        'dec_deg': dec_deg,
        'lat_deg': lat_deg,
        'lon_deg': lon_deg,
        'r_eci': r_eci_agg,
        'fit_rms_deg': float(np.mean(fit_values)),
        'fit_max_deg': float(np.max(fit_values)),
        'triangle_pair_rms_deg': float(np.mean(pair_values)),
        'spread_deg': float(np.mean(spread_values)),
        'timestamp_utc': now_utc.isoformat(),
        'used_hips': best_h['solution']['used_hips'],
        'used_hypotheses': len(hypotheses_in),
        'total_hypotheses': len(hypotheses),
        'max_triplet_support': int(max(hips_support.values()) if hips_support else 1),
        'best_triangle_id': best_h['triangle_id'],
    }


def generate_triangles_from_stars(stars, top_n=10, min_separation_px=12.0):
    brightest_stars = select_spread_bright_stars(stars, top_n=top_n, min_separation_px=min_separation_px)

    triangles_data = []

    for star_combo in combinations(brightest_stars, 3):
        s1, s2, s3 = star_combo

        d1 = calculate_distance(s1.center, s2.center)
        d2 = calculate_distance(s2.center, s3.center)
        d3 = calculate_distance(s1.center, s3.center)

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
        orientation = orientation_sign_2d(s1.center, s2.center, s3.center)

        triangle_id = f"tri_{s1.id}_{s2.id}_{s3.id}"

        triangles_data.append({
            'triangle_id': triangle_id,
            'star_ids': (s1.id, s2.id, s3.id),
            'ratio1': ratio1,
            'ratio2': ratio2,
            'ratio3': ratio3,
            'orientation': orientation
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

        triangles = generate_triangles_from_stars(stars, top_n=8)

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
            matches = star_db.find_best_match(
                t['ratio1'],
                t['ratio2'],
                t['ratio3'],
                tolerance=0.015,
                fov_x_deg=60.0,
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
            per_triangle_k=20
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
                best_solution = estimate_pointing_multi_triangle(
                    star_db=star_db,
                    stars=stars,
                    image_path="stars.png",
                    triangles=triangles,
                    fov_x_deg=60.0,
                    tolerance=0.015,
                    per_triangle_k=3,
                    max_fit_rms_deg=6.0,
                )

                if best_solution is not None:
                    print("\nCoordinate-transform solution (improved fit):")
                    print(
                        f"  UTC: {best_solution['timestamp_utc']} | "
                        f"RA={best_solution['ra_deg']:.4f}°, Dec={best_solution['dec_deg']:.4f}°"
                    )
                    print(
                        f"  Estimated Earth point: lat={best_solution['lat_deg']:.4f}°, "
                        f"lon={best_solution['lon_deg']:.4f}°"
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
