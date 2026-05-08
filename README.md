# Data-Fitting
part 2 - Data fitting for lab Math Application 

## Project layout

- `Group_01/part1/ols_toolkit.py`: NumPy-only implementations for OLS, hat matrix, ridge, lasso, metrics, and k-fold CV.
- `Group_01/part2/data_pipeline.py`: Leakage-safe `DataPipeline` with training-only fit statistics.
- `tests/`: `unittest` coverage for Part 1 and Part 2 components.

## Run tests

```bash
python -m unittest discover -s tests -v
```
