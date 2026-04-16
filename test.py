import psycopg2
from config_db import DB_CONFIG

def test_spatial_query():
    try:
        conn = psycopg2.connect(**DB_CONFIG)
    except psycopg2.OperationalError as e:
        print(f"Database connection error: {e}")
        return

    cursor = conn.cursor()

    target_ra_hours = 6.0
    target_ra = target_ra_hours * 15.0
    target_dec = 10.0
    radius_degrees = 1.0 

    print(
        f"Searching for stars within {radius_degrees}° of point "
        f"(RA:{target_ra}° / {target_ra_hours}h, Dec:{target_dec})..."
    )

   
    query = """
        SELECT hip_id, magnitude, ra, dec
        FROM star_catalog
        WHERE ST_DWithin(
            geom,
            ST_SetSRID(ST_MakePoint(%s, %s), 4326),
            %s
        )
        ORDER BY magnitude ASC -- Сортуємо: спочатку найяскравіші
        LIMIT 5; -- Беремо лише 5 найяскравіших
    """

    try:
        cursor.execute(query, (target_ra, target_dec, radius_degrees))
        stars = cursor.fetchall()
    except Exception as e:
        print(f"Query execution error: {e}")
        cursor.close()
        conn.close()
        return

    if not stars:
        print("No stars found.")
    else:
        print(f"Found {len(stars)} stars:")
        for star in stars:
            hip_id, mag, ra, dec = star
            print(f"  Star HIP {hip_id} | Magnitude: {mag} | RA: {ra}, Dec: {dec}")

    cursor.close()
    conn.close()

if __name__ == "__main__":
    test_spatial_query()