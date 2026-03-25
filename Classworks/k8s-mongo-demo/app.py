from flask import Flask, request
from pymongo import MongoClient
import os

app = Flask(__name__)

mongo_user = os.environ.get("MONGO_USERNAME", "root")
mongo_password = os.environ.get("MONGO_PASSWORD", "example")
mongo_host = os.environ.get("MONGO_HOST", "mongo-service")
mongo_port = os.environ.get("MONGO_PORT", "27017")
mongo_db = os.environ.get("MONGO_DB", "demo_db")

mongo_uri = f"mongodb://{mongo_user}:{mongo_password}@{mongo_host}:{mongo_port}/"

client = MongoClient(mongo_uri)
db = client[mongo_db]
collection = db["settings"]


def get_current_value():
    doc = collection.find_one({"_id": "current"})

    if doc:
        return doc.get("value", "")
    return ""


@app.route("/", methods=["GET"])
def index():
    current_value = get_current_value()
    return f"""
<html>
<body>
<h2>Simple Kubernetes + Mongo Demo</h2>
<form action="/update" method="post">
<label>Enter a value:</label>
<input type="text" name="value" />
<button type="submit">Save</button>
</form>
<p><strong>Current saved value:</strong> {current_value}</p>
</body>
</html>
"""


@app.route("/update", methods=["POST"])
def update():
    new_value = request.form.get("value", "")
    collection.update_one(
        {"_id": "current"},
        {"$set": {"value": new_value}},
        upsert=True
    )
    return f"""
<html>
<body>
<p>Saved value: <strong>{new_value}</strong></p>
<a href="/">Go back</a>
</body>
</html>
"""


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)