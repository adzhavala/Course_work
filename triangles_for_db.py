import psycopg2
import math
from itertools import combinations
from config_db import DB_CONFIG


def orientation_sign(p1, p2, p3):
    return math.copysign(1.0, (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0]))


def angular_distance(ra1_deg, dec1_deg, ra2_deg, dec2_deg):
    ra1_rad = math.radians(ra1_deg)
    ra2_rad = math.radians(ra2_deg)

    dec1_rad = math.radians(dec1_deg)
    dec2_rad = math.radians(dec2_deg)

    term = math.sin(dec1_rad) * math.sin(dec2_rad) + \
           math.cos(dec1_rad) * math.cos(dec2_rad) * math.cos(ra1_rad - ra2_rad)

    term = max(-1.0, min(1.0, term))
    return math.degrees(math.acos(term))


def build_triangle_database(
    max_stars=1500,
    max_magnitude=4.5,
    min_side_deg=0.5,
    max_side_deg=50.0,
    min_angle_deg=10.0,
    max_angle_deg=170.0,
    insert_batch_size=5000,
):
    try:
        conn = psycopg2.connect(**DB_CONFIG)
    except psycopg2.OperationalError as e:
        print(f"Database connection error: {e}")
        print("Check config_db.py and PostgreSQL is running")
        return

    cursor = conn.cursor()

    try:
        print("Clearing old triangle database...")
        cursor.execute("TRUNCATE TABLE star_triangles RESTART IDENTITY;")

        print("Fetching stars from database...")
        cursor.execute(
            """
            SELECT hip_id, ra, dec
            FROM star_catalog
            WHERE magnitude <= %s
            ORDER BY magnitude ASC
            LIMIT %s;
            """,
            (max_magnitude, max_stars),
        )
        stars = cursor.fetchall()
        print(f"Retrieved {len(stars)} brightest stars.")

        if stars:
            max_ra = max(s[1] for s in stars)
            if max_ra <= 24.5:
                print("RA in hours detected. Converting to degrees for calculations.")
                stars = [(hip, ra * 15.0, dec) for hip, ra, dec in stars]

        if not stars:
            print("No stars in database. Run load_stars.py first")
            cursor.close()
            conn.close()
            return
    except Exception as e:
        print(f"Database query error: {e}")
        cursor.close()
        conn.close()
        return

    total_combinations = math.comb(len(stars), 3)
    print(f"Processing {total_combinations} combinations (this will take a few seconds)...")

    inserted_count = 0
    batch = []

    for i, combo in enumerate(combinations(stars, 3)):
        if i % 200000 == 0 and i > 0:
            percent = (i / total_combinations) * 100
            print(f"Processed {i} combinations ({percent:.1f}%)...")

        s1, s2, s3 = combo

        d1 = angular_distance(s1[1], s1[2], s2[1], s2[2])
        d2 = angular_distance(s2[1], s2[2], s3[1], s3[2])
        d3 = angular_distance(s1[1], s1[2], s3[1], s3[2])

        sides = sorted([d1, d2, d3])
        a, b, c = sides[0], sides[1], sides[2]

        if a < min_side_deg or c > max_side_deg:
            continue

        cos_gamma = (a**2 + b**2 - c**2) / (2 * a * b)
        cos_gamma = max(-1.0, min(1.0, cos_gamma))
        max_angle = math.degrees(math.acos(cos_gamma))

        cos_alpha = (b**2 + c**2 - a**2) / (2 * b * c)
        cos_alpha = max(-1.0, min(1.0, cos_alpha))
        min_angle = math.degrees(math.acos(cos_alpha))

        if min_angle < min_angle_deg or max_angle > max_angle_deg:
            continue

        ratio1 = b / a
        ratio2 = c / a
        ratio3 = b / c

        dec_mean = math.radians((s1[2] + s2[2] + s3[2]) / 3.0)
        p1 = (s1[1] * math.cos(dec_mean), s1[2])
        p2 = (s2[1] * math.cos(dec_mean), s2[2])
        p3 = (s3[1] * math.cos(dec_mean), s3[2])
        orient = orientation_sign(p1, p2, p3)

        batch.append((s1[0], s2[0], s3[0], ratio1, ratio2, ratio3, orient, c))

        if len(batch) >= insert_batch_size:
            try:
                try:
                    insert_query = """
                        INSERT INTO star_triangles (star1_hip, star2_hip, star3_hip, ratio1, ratio2, ratio3, orientation, max_side_deg)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    """
                    cursor.executemany(insert_query, batch)
                except Exception:
                    insert_query_legacy = """
                        INSERT INTO star_triangles (star1_hip, star2_hip, star3_hip, ratio1, ratio2, max_side_deg)
                        VALUES (%s, %s, %s, %s, %s, %s)
                    """
                    legacy_payload = [(t[0], t[1], t[2], t[3], t[4], t[-1]) for t in batch]
                    cursor.executemany(insert_query_legacy, legacy_payload)

                conn.commit()
                inserted_count += len(batch)
                batch.clear()
            except Exception as e:
                print(f"Write error: {e}")
                conn.rollback()
                cursor.close()
                conn.close()
                return

    if batch:
        try:
            try:
                insert_query = """
                    INSERT INTO star_triangles (star1_hip, star2_hip, star3_hip, ratio1, ratio2, ratio3, orientation, max_side_deg)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """
                cursor.executemany(insert_query, batch)
            except Exception:
                insert_query_legacy = """
                    INSERT INTO star_triangles (star1_hip, star2_hip, star3_hip, ratio1, ratio2, max_side_deg)
                    VALUES (%s, %s, %s, %s, %s, %s)
                """
                legacy_payload = [(t[0], t[1], t[2], t[3], t[4], t[-1]) for t in batch]
                cursor.executemany(insert_query_legacy, legacy_payload)

            conn.commit()
            inserted_count += len(batch)
            batch.clear()
        except Exception as e:
            print(f"Write error: {e}")
            conn.rollback()
            cursor.close()
            conn.close()
            return

    print(f"Generation complete! Selected {inserted_count} ideal triangles for database.")
    print("Triangle database updated.")

    cursor.close()
    conn.close()


if __name__ == "__main__":
    build_triangle_database()