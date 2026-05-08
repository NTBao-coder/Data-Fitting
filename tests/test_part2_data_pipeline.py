import unittest

import numpy as np

from Group_01.part2.data_pipeline import DataPipeline


class TestDataPipeline(unittest.TestCase):
    def setUp(self) -> None:
        self.X_train = np.array(
            [
                [1.0, "A", 10.0],
                [2.0, "B", None],
                [None, "A", 30.0],
                [4.0, "", 40.0],
            ],
            dtype=object,
        )
        self.X_test = np.array(
            [
                [None, "B", 20.0],
                [5.0, "C", None],
            ],
            dtype=object,
        )
        self.pipeline = DataPipeline(numeric_indices=(0, 2), categorical_indices=(1,))

    def test_fit_transform_produces_expected_width(self) -> None:
        transformed = self.pipeline.fit_transform(self.X_train)
        self.assertEqual(transformed.shape[0], self.X_train.shape[0])
        self.assertEqual(transformed.shape[1], 2 + 2)  # 2 numeric + 2 categories (A, B)

    def test_transform_uses_train_statistics_without_leakage(self) -> None:
        self.pipeline.fit(self.X_train)
        transformed_test = self.pipeline.transform(self.X_test)
        self.assertEqual(transformed_test.shape[1], 4)
        # unseen category C should not create new columns and should encode to zeros for category block
        np.testing.assert_array_equal(transformed_test[1, 2:], np.array([0.0, 0.0]))

    def test_transform_before_fit_raises(self) -> None:
        with self.assertRaises(RuntimeError):
            self.pipeline.transform(self.X_test)

    def test_empty_feature_configuration_returns_empty_matrix(self) -> None:
        pipeline = DataPipeline(numeric_indices=(), categorical_indices=())
        transformed = pipeline.fit_transform(self.X_train)
        self.assertEqual(transformed.shape, (self.X_train.shape[0], 0))


if __name__ == "__main__":
    unittest.main()
