import sqlite3
import random

# Connect to the database
conn = sqlite3.connect("./NLP/db_stack.db")
cursor = conn.cursor()

# Fill the new column with random values
cursor.execute("UPDATE Stackoverflow SET Toxic = (?)", (random.randint(0,1),))
conn.commit()

# Close the connection
conn.close()