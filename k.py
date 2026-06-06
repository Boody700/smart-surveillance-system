import sqlite3
conn = sqlite3.connect("database/sentinel.db")
conn.execute("ALTER TABLE events ADD COLUMN bbox_x1 INTEGER")
conn.execute("ALTER TABLE events ADD COLUMN bbox_y1 INTEGER")
conn.execute("ALTER TABLE events ADD COLUMN bbox_x2 INTEGER")
conn.execute("ALTER TABLE events ADD COLUMN bbox_y2 INTEGER")
conn.commit()
conn.close()
print("columns added")