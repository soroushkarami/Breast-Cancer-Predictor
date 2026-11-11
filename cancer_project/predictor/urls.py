from django.urls import path
from .views import predict_view
from .views import export_history_csv

urlpatterns = [
    path("", predict_view, name="predict"),
    path('export-history/', export_history_csv, name='export_history')
]