
import json
import os
import unittest

import joblib
import pandas as pd


class TestMLPipeline(unittest.TestCase):

    def test_model_file_exists(self):
        self.assertTrue(
            os.path.exists("student_result_model.pkl"),
            "Trained model file is missing"
        )

    def test_metrics_file_exists(self):
        self.assertTrue(
            os.path.exists("metrics.json"),
            "Metrics file is missing"
        )

    def test_dataset_file_exists(self):
        self.assertTrue(
            os.path.exists("student_results.csv"),
            "Dataset file is missing"
        )

    def test_metrics_are_valid(self):
        with open("metrics.json", "r") as file:
            metrics = json.load(file)

        self.assertIn("accuracy", metrics)
        self.assertGreaterEqual(metrics["accuracy"], 0.0)
        self.assertLessEqual(metrics["accuracy"], 1.0)

    def test_dataset_has_required_columns(self):
        data = pd.read_csv("student_results.csv")

        expected_columns = {
            "study_hours",
            "attendance",
            "previous_score",
            "result"
        }

        self.assertTrue(expected_columns.issubset(data.columns))
        self.assertGreater(len(data), 0)

    def test_model_can_make_predictions(self):
        model = joblib.load("student_result_model.pkl")

        sample = pd.DataFrame({
            "study_hours": [6],
            "attendance": [75],
            "previous_score": [70]
        })

        prediction = model.predict(sample)

        self.assertEqual(len(prediction), 1)
        self.assertIn(prediction[0], [0, 1])


if __name__ == "__main__":
    unittest.main()
