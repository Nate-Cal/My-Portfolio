import csv
import os

# Path to CSV file
csv_path = os.path.expanduser('~/LUC/COMP379/HW1/titanic/test.csv')

def predict_survival(csv_path):
    # Initialize Results
    results = []
    # Parse the CSV file
    with open(csv_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile, delimiter=',')
        for row in reader:
            # Extract relevant features with default values for missing data
            try:
                pclass = int(row.get("Pclass", 0) or 0)
                fare = float(row.get("Fare", 0) or 0)
                sex = str(row.get("Sex", 0) or "")
                age = float(row.get("Age", 0) or 0)
                sibsp = int(row.get("SibSp", 0) or 0)
                parch = int(row.get("Parch", 0) or 0)
            except ValueError:
                pclass, fare, age, sibsp, parch = 0, 0, 0, 0, 0
                sex = ""
            # Calculate the survival prediction based on the relevant features
            survival_rate = (1 if pclass == 1 else 0) + (1 if sibsp == 0 else 0) + (1 if parch == 0 else 0)  + (1 if age >= 18 else 0) + (1 if fare >= 25 else 0) + (1 if sex.lower() == "female" else 0)
            survived = survival_rate >= 4
            results.append((row["PassengerId"], survived))
    return results

if __name__ == "__main__":
    predictions = predict_survival(csv_path)
    for pid, survived in predictions:
        print(f"PassengerId: {pid}, Survived: {survived}")