    # To implement database:
from django.db import models

class Prediction(models.Model):
    radius_mean = models.FloatField()
    texture_mean = models.FloatField()
    concave_points_mean = models.FloatField()
    radius_worst = models.FloatField()
    perimeter_worst = models.FloatField()
    area_worst = models.FloatField()
    concave_points_worst = models.FloatField()

    prediction = models.IntegerField()
    probability = models.FloatField()
    model_used = models.CharField(max_length=20, default='logistic')
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f'Num {self.id}  |  {self.prediction}  |  Confidence: {self.probability:.4f}'