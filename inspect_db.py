
import sqlite3

DB_FILE = "user_data.db"

try:
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    print("Conexiune la bd efectuată cu succes.")

    print("\Tabelele din bd sunt:")
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    for table in tables:
        print(f"- {table[0]}")

    print("\Conținutul tabelului 'users':")
    cursor.execute("SELECT * FROM users")
    users = cursor.fetchall()
    for user in users:
        print(f"ID: {user[0]}, Username: {user[1]}, Hashed Password: {user[2]}")

except sqlite3.Error as e:
    print(f"Database error: {e}")
finally:
    if conn:
        conn.close()
        print("\nDatabase connection closed.")