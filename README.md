# Breast Cancer Predictor

Django app serving three classifiers (logistic regression implemented from scratch in NumPy, plus KNN and SVM from scikit-learn) on the Wisconsin Breast Cancer dataset.

## Run

    pip install -r requirements.txt
    python manage.py migrate
    python manage.py runserver

Open http://127.0.0.1:8000

## Structure

- `cancer_project/predictor/` — Django app (views, models, templates, model wrappers)
- `cancer_project/predictor/ml_models/` — trained model artifacts
- `the model/` — training notebook and dataset

## TODO

- Move SECRET_KEY to environment variables
- Add unit tests
- Fix bugs in `views.py` and `models.py`
- Convert notebook absolute paths to relative

