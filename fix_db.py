import psycopg2
from config_db import DB_CONFIG

try:
    conn = psycopg2.connect(**DB_CONFIG)
    conn.autocommit = True
    cursor = conn.cursor()
    
    cursor.execute("ALTER TABLE star_triangles DROP COLUMN IF EXISTS orientation;")
    print("✅ Колонку 'orientation' успішно назавжди видалено з бази даних!")
    
    cursor.close()
    conn.close()
except Exception as e:
    print(f"Помилка: {e}")