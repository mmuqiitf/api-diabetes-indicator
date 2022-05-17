from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('submit-survey', views.submit_survey, name='submit_survey'),
    path('submit-survey-diabetes', views.submit_survey_diabetes, name='submit_survey_diabetes'),
    path('datasets', views.datasets, name='datasets'),
    path('datasets-diabetes', views.datasets_diabetes, name='datasets_diabetes'),
    path('age-frequency', views.age_frequency, name='age_frequency'),
]