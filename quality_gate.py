
import json
import sys

METRICS_FILE = "metrics.json"
ACCURACY_THRESHOLD = 0.85

try:
    with open(METRICS_FILE, "r") as file:
        metrics = json.load(file)

    accuracy = metrics.get("accuracy")

    if accuracy is None:
        print("QUALITY GATE FAILED: Accuracy metric is missing.")
        sys.exit(1)

    print(f"Model accuracy: {accuracy:.2%}")
    print(f"Required accuracy: {ACCURACY_THRESHOLD:.2%}")

    if accuracy >= ACCURACY_THRESHOLD:
        print("QUALITY GATE PASSED")
        sys.exit(0)
    else:
        print("QUALITY GATE FAILED: Accuracy is below 85%.")
        sys.exit(1)

except FileNotFoundError:
    print("QUALITY GATE FAILED: metrics.json was not found.")
    sys.exit(1)

except (json.JSONDecodeError, TypeError, ValueError) as error:
    print(f"QUALITY GATE FAILED: Invalid metrics file: {error}")
    sys.exit(1)
