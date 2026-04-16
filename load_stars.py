import pandas as pd
import psycopg2
from config_db import DB_CONFIG

def load_stars_to_db():
    try:
        conn = psycopg2.connect(**DB_CONFIG)
    except psycopg2.OperationalError as e:
        print(f"Database connection error: {e}")
        return

    cursor = conn.cursor()

    try:
        df = pd.read_csv('hyg_v42.csv')
    except FileNotFoundError:
        print("File not found")
        cursor.close()
        conn.close()
        return
    except Exception as e:
        print(f"CSV reading error: {e}")
        cursor.close()
        conn.close()
        return


    df_bright = df[df['mag'] < 6.0].copy()


    df_bright['hip'] = df_bright['hip'].fillna(0)

    for index, row in df_bright.iterrows():
        hip = int(row['hip'])
        mag = float(row['mag'])
        ra_hours = float(row['ra'])
        ra_deg = ra_hours * 15.0
        dec = float(row['dec'])
        # sql запит із функцією postgis для створення точки
        insert_query = """
            INSERT INTO star_catalog (hip_id, magnitude, ra, dec, geom)
            VALUES (%s, %s, %s, %s, ST_SetSRID(ST_MakePoint(%s, %s), 4326));
        """
        cursor.execute(insert_query, (hip, mag, ra_deg, dec, ra_deg, dec))

    conn.commit()
    cursor.close()
    conn.close()
    print("All stars loaded into PostGIS.")


if __name__ == "__main__":
    load_stars_to_db()