from django.shortcuts import render
from .ml_models.log_reg_model.model import BreastCancerLogisticModel
from .ml_models.knn_model.model import BreastCancerKNNModel
from .ml_models.svm_model.model import BreastCancerSVMModel
from .models import Prediction

log_model = BreastCancerLogisticModel()
knn_model = BreastCancerKNNModel()
svm_model = BreastCancerSVMModel()
# loads the models once the server starts

def predict_view(request):
    results = []
    error = None
    if request.method == 'POST':
        try:
            features = [
                float(request.POST.get('f1')),
                float(request.POST.get('f2')),
                float(request.POST.get('f3')),
                float(request.POST.get('f4')),
                float(request.POST.get('f5')),
                float(request.POST.get('f6')),
                float(request.POST.get('f7'))
            ]
            ranges = [
                (6, 30),    # radius_mean
                (9, 40),    # texture_mean
                (0, 0.3),    # concave_points_mean
                (7, 40),    # radius_worst
                (50, 250),    # perimeter_worst
                (200, 3000),    # area_worst
                (0, 0.35)    # concave_points_worst
            ]
            for val, (low, high) in zip(features, ranges):
                if not (low <= val <= high):
                    error = 'Entered values are not in the realistic ranges'
                    break

            else:
                model_choice = request.POST.get('model_choice', 'logistic')
                compare = request.POST.get('comparison')

                models = {
                    "logistic": ("Logistic Regression", log_model),
                    "knn": ("KNN", knn_model),
                    "svm": ("SVM", svm_model)
                }

                # If comparison checkbox is checked → run all models
                if compare:
                    selected_models = models.items()
                else:
                    selected_models = [(model_choice, models[model_choice])]

                for key, (name, model) in selected_models:

                    prediction, prob = model.predict(features= features)

                    risk = 'LOW'
                    if prediction == 1:
                        confidence = prob * 100
                        if prob >= 0.85:
                            risk = 'HIGH'
                        elif 0.6 < prob < 0.85:
                            risk = 'MODERATE'
                    else:
                        confidence = (1 - prob) * 100

                    # Save prediction in DB
                    Prediction.objects.create(
                        radius_mean=features[0],
                        texture_mean=features[1],
                        concave_points_mean=features[2],
                        radius_worst=features[3],
                        perimeter_worst=features[4],
                        area_worst=features[5],
                        concave_points_worst=features[6],
                        prediction=prediction,
                        probability=confidence,
                        model_used=key
                    )

                    if prediction == 1:
                        text = f'{name}: Malignant | Confidence: {confidence:.2f}% | Risk: {risk}'
                    else:
                        text = f'{name}: Benign | Confidence: {confidence:.2f}%'

                    results.append(text)

        except ValueError:
            error = 'Entered values must be numeric'


    # to show history in the main page
    history = Prediction.objects.all().order_by('-created_at')[:10]
    # '-' before created_at means descending order instead of default which is ascending
    # [:10] tells Django to return only the latest 10 records

    return render(request, "predictor/form.html", {
        "results": results,
        'error': error,
        'history': history
    })
    # Inside form.html, the variable {{ result }} will contain whatever the view calculated


# export csv feature
import csv
from django.http import HttpResponse

def export_history_csv(response):
    response = HttpResponse(content_type='text/csv')
    response['Content-Disposition'] = 'attachment; filename="prediction_history.csv"'

    writer = csv.writer(response)

    # headers
    writer.writerow([
        'Created at', 'Radius Mean',
        'Texture Mean', 'Concave Point Mean',
        'Radius Worst', 'Perimeter Worst',
        'Area Worst', 'Concave Points Worst',
        'Prediction',
        'Confidence',
        'Model'
    ])

    # Rows
    for p in Prediction.objects.all().order_by('-created_at'):
        the_model = 'Logistic Regression'
        if p.model_used == 'knn':
            the_model = 'KNN'
        elif p.model_used == 'svm':
            the_model = 'SVM'
        writer.writerow([
            p.created_at, p.radius_mean,
            p.texture_mean, p.concave_points_mean,
            p.radius_worst, p.perimeter_worst,
            p.area_worst, p.concave_points_worst,
            "Malignant" if p.prediction == 1 else "Benign",
            p.probability,
            the_model
        ])

    return response