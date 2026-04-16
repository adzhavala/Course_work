import cv2
import numpy as np
import math
from itertools import combinations


class Star:
    def __init__(self, star_id, center, radius, brightness):
        self.id = star_id
        self.center = center
        self.radius = radius
        self.brightness = brightness

    def __repr__(self):
        return f"Star(ID={self.id}, Center={self.center}, R={self.radius:.2f}, Bright={self.brightness:.2f})"


def find_stars_advanced(path, threshold_factor=2.5, min_area=2, max_area=500):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Failed to open image")

    blurred = cv2.GaussianBlur(img, (3, 3), 0)
    mean, std = cv2.meanStdDev(blurred)

    mean_val = mean[0][0]
    std_val = std[0][0]
    T = mean_val + threshold_factor * std_val

    _, binary = cv2.threshold(blurred, T, 255, cv2.THRESH_BINARY)

    kernel = np.ones((2, 2), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary)

    stars_data = []

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]

        if area < min_area or area > max_area:
            continue

        cx = float(centroids[i][0])
        cy = float(centroids[i][1])

        star_mask = (labels == i).astype(np.uint8) * 255

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


def generate_triangles_from_stars(stars, top_n=10):
    brightest_stars = sorted(stars, key=lambda s: s.brightness, reverse=True)[:top_n]

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
                t['orientation'],
                tolerance=0.015
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
