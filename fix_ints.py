import sqlite3

db_file = "experiments.sqlite"

db_conn = sqlite3.connect(db_file)
cursor = db_conn.cursor()

select = "SELECT n_constraints, id FROM experiments"
update = "UPDATE experiments SET n_constraints=? WHERE id=?"

cursor.execute("BEGIN DEFERRED TRANSACTION")
for row in cursor.execute(select).fetchall():
    cursor.execute(update, [int.from_bytes(r, byteorder='little') if isinstance(r, bytes) else r for r in row])

db_conn.commit()

