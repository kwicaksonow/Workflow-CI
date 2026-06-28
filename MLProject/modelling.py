import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import mlflow
import mlflow.sklearn

df = pd.read_csv("titanic_preprocessing.csv")
X = df.drop(columns=["Survived"])
y = df["Survived"]

with mlflow.start_run() as run:
    # Latih model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)

    mlflow.sklearn.log_model(model, "model")

    with open("run_id.txt", "w") as f:
        f.write(run.info.run_id)

    print(f"Model disimpan dengan ID: {run.info.run_id}")
