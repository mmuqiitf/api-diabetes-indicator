from django.shortcuts import render
from django.http.response import HttpResponse, JsonResponse
from rest_framework.decorators import api_view

import pickle
import os
import json
import numpy as np
import pandas as pd
import re

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

filename = 'finalized_model.sav'
filename_svm = 'finalized_model_svm.sav'
# Load the model from the pickle file using absolut path
model = pickle.load(open(os.path.join(os.path.dirname(__file__), filename), 'rb'))
model_rf = pickle.load(open(os.path.join(os.path.dirname(__file__), filename_svm), 'rb'))

# Load datasets heart_2020_cleaned.csv
df = pd.read_csv(os.path.join(os.path.dirname(__file__), 'heart_2020_cleaned.csv'))

# Load datasets diabetes_012_health_indicators_BRFSS2015.csv
df_diabetes = pd.read_csv(os.path.join(os.path.dirname(__file__), 'diabetes_012_health_indicators_BRFSS2015.csv'))

# Encode age category column
encode_AgeCategory = {'55-59':57, '80 or older':80, '65-69':67,
                      '75-79':77,'40-44':42,'70-74':72,'60-64':62,
                      '50-54':52,'45-49':47,'18-24':21,'35-39':37,
                      '30-34':32,'25-29':27}
df['AgeCategory'] = df['AgeCategory'].apply(lambda x: encode_AgeCategory[x])
df['AgeCategory'] = df['AgeCategory'].astype('float')


def index(response):
    return HttpResponse("Hello, world. You're at the main index.")

@api_view(['GET'])
def datasets(request):
    heart_disease_label = np.array(df['HeartDisease'].value_counts().index)
    heart_disease_value = [x for x in df['HeartDisease'].value_counts()]

    smoking_label = np.array(df['Smoking'].value_counts().index)
    smoking_value = [x for x in df['Smoking'].value_counts()]
    
    alcohol_drinking_label = np.array(df['AlcoholDrinking'].value_counts().index)
    alcohol_drinking_value = [x for x in df['AlcoholDrinking'].value_counts()]
    
    stroke_label = np.array(df['Stroke'].value_counts().index)
    stroke_value = [x for x in df['Stroke'].value_counts()]

    # DiffWalking
    diff_walking_label = np.array(df['DiffWalking'].value_counts().index)
    diff_walking_value = [x for x in df['DiffWalking'].value_counts()]

    # Sex
    sex_label = np.array(df['Sex'].value_counts().index)
    sex_value = [x for x in df['Sex'].value_counts()]

    # Physical Activity
    physical_act_label = np.array(df['PhysicalActivity'].value_counts().index)
    physical_act_value = [x for x in df['PhysicalActivity'].value_counts()]

    # Diabetic
    diabetic_label = np.array(df['Diabetic'].value_counts().index)
    diabetic_value = [x for x in df['Diabetic'].value_counts()]

    return JsonResponse({
        "heart_disease": {
            "label": heart_disease_label.tolist(), 
            "value": heart_disease_value, 
            "title": re.sub('([A-Z])', r' \1', "Heart Disease").title()
        },
        "smoking": {
            "label": smoking_label.tolist(),
            "value": smoking_value,
            "title": re.sub('([A-Z])', r' \1', "Smoking").title()
        },
        "alcohol_drinking": {
            "label": alcohol_drinking_label.tolist(),
            "value": alcohol_drinking_value,
            "title": re.sub('([A-Z])', r' \1', "Alcohol Drinking").title()
        },
        "stroke": {
            "label": stroke_label.tolist(),
            "value": stroke_value,
            "title": re.sub('([A-Z])', r' \1', "Stroke").title()
        },
        "diff_walking": {
            "label": diff_walking_label.tolist(),
            "value": diff_walking_value,
            "title": re.sub('([A-Z])', r' \1', "Difficult Walk").title()
        },
        "sex": {
            "label": sex_label.tolist(),
            "value": sex_value,
            "title": re.sub('([A-Z])', r' \1', "Sex").title()
        },
        "physical_act": {
            "label": physical_act_label.tolist(),
            "value": physical_act_value,
            "title": re.sub('([A-Z])', r' \1', "Physical Activity").title()
        },
        "diabetic": {
            "label": diabetic_label.tolist(),
            "value": diabetic_value,
            "title": re.sub('([A-Z])', r' \1', "Diabetic").title()
        },
    }, safe=True)


@api_view(['GET'])
def age_frequency(request):
    heart_disease = df[df["HeartDisease"]=='Yes']["AgeCategory"].to_list()
    kidney_disease = df[df["KidneyDisease"]=='Yes']["AgeCategory"].to_list()
    skin_cancer = df[df["SkinCancer"]=='Yes']["AgeCategory"].to_list()
    return JsonResponse({"heart_disease": heart_disease, "kidney_disease": kidney_disease, "skin_cancer": skin_cancer}, safe=True)


@api_view(['GET', 'POST'])
def submit_survey(request):
    surveyData = request.data['surveyData']
    # bmi describe
    bmi_describe = {
        'min': 12.020000,
        'mean': 28.325399,
        'max': 94.850000,
    }

    # physical health describe
    physical_health_describe =  {
        'min': 0.000000,
        'mean': 3.371710,
        'max': 30.00000,
    }

    # mental health describe
    mental_health_describe = {
        'min': 0.000000,
        'mean': 3.898366,
        'max': 30.00000,
    }

    # age category describe
    age_category_describe = {
        'min': 21.000000,
        'mean': 54.355759,
        'max': 80.000000,
    }

    # sleep time
    sleep_time_describe = {
        'min': 1.000000,
        'mean': 7.097075,
        'max': 24.000000,
    }

    # bmi = surveyData['weight'] / (surveyData['height'] * surveyData['height'])
    bmi = float(surveyData[0]['answerByUser']['weight']) / (
                float(surveyData[0]['answerByUser']['height']) * float(surveyData[0]['answerByUser']['height']))
    bmiAnswer = bmi / bmi_describe['max']
    # smoking
    smoking = surveyData[1]['answerByUser']
    # alcohol
    alcohol = surveyData[2]['answerByUser']
    # stroke
    stroke = surveyData[3]['answerByUser']
    # physical health
    physicalHealth = int(surveyData[4]['answerByUser'])
    physicalHealthAnswer = physicalHealth / physical_health_describe['max']
    # mental health
    mentalHealth = int(surveyData[5]['answerByUser'])
    mentalHealthAnswer = mentalHealth / mental_health_describe['max']
    # diff walking
    diffWalking = surveyData[6]['answerByUser']
    # sex
    sex = surveyData[7]['answerByUser']
    # age category
    ageCategory = int(surveyData[8]['answerByUser'])
    ageCategoryAnswer = ageCategory / age_category_describe['max']
    # physical activity
    physicalAct = surveyData[9]['answerByUser']
    # sleep time
    sleepTime = int(surveyData[10]['answerByUser'])
    sleepTimeAnswer = sleepTime / sleep_time_describe['max']
    # asthma
    asthma = surveyData[11]['answerByUser']
    # kidney disease
    kidneyDisease = surveyData[12]['answerByUser']
    # skin cancer
    skinCancer = surveyData[13]['answerByUser']
    # race/ethnicity
    # american, asian, black, hispanic, other, white
    race = surveyData[14]['answerByUser']
    raceAnswer = [1,0,0,0,0,0] if race == 'american' else [0,1,0,0,0,0] if race == 'asian' else [0,0,1,0,0,0] if race == 'black' else [0,0,0,1,0,0] if race == 'hispanic' else [0,0,0,0,1,0] if race == 'other' else [0,0,0,0,0,1] if race == 'white' else [0,0,0,0,0,0]
    # diabetic
    diabetic = surveyData[15]['answerByUser']
    diabeticAnswer = [1,0,0,0] if diabetic == 'no' else [0,1,0,0] if diabetic == 'borderline-diabetes' else [0,0,1,0] if diabetic == 'yes' else [0,0,0,1] if diabetic == 'yes-pregnancy' else [0,0,0,0]
    # general health
    genHealth = surveyData[16]['answerByUser']
    genHealthAnswer = [1,0,0,0,0] if genHealth == 'poor' else [0,1,0,0,0] if genHealth == 'fair' else [0,0,1,0,0] if genHealth == 'good' else [0,0,0,1,0] if genHealth == 'very-good' else [0,0,0,0,1] if genHealth == 'excellent' else [0,0,0,0,0]

    # data predict
    dataPredict = [bmiAnswer, smoking, alcohol, stroke, physicalHealthAnswer, mentalHealthAnswer, diffWalking, sex, ageCategoryAnswer, physicalAct, sleepTimeAnswer, asthma, kidneyDisease, skinCancer] + raceAnswer + diabeticAnswer + genHealthAnswer
    # predict from model
    result_predict = model.predict([dataPredict])
    response = JsonResponse({'message': 'Hello, world. You\'re at the main index.', 'bmi': bmiAnswer, 'age_category': ageCategoryAnswer, 'physical_health': physicalHealthAnswer,
                             'mental_health': mentalHealthAnswer, 'sleep_time': sleepTimeAnswer, 'race': raceAnswer, 'gen_health': genHealthAnswer, 'diabetic': diabeticAnswer, 'data_predict': dataPredict, 'predict': json.dumps(result_predict[0], cls=NpEncoder)
                             })
    return response

@api_view(['GET', 'POST'])
def submit_survey_diabetes(request):
    surveyData = request.data['surveyData']
    # high bp
    high_bp = int(surveyData[0]['answerByUser'])

    # high cholesterol
    high_cholesterol = int(surveyData[1]['answerByUser'])

    # cholesterol check 5 years
    cholesterol_check = int(surveyData[2]['answerByUser'])

    # bmi
    bmi = float(surveyData[3]['answerByUser']['weight']) / (
            float(surveyData[3]['answerByUser']['height']) * float(surveyData[3]['answerByUser']['height']))

    # smoking
    smoking = int(surveyData[4]['answerByUser'])

    # stroke
    stroke = int(surveyData[5]['answerByUser'])

    # heart disease or attack
    heart_disease = int(surveyData[6]['answerByUser'])

    # physical activity
    physical_activity = int(surveyData[7]['answerByUser'])

    # fruits
    fruits = int(surveyData[8]['answerByUser'])

    # veggies
    veggies = int(surveyData[9]['answerByUser'])

    # heavy alcohol consump
    heavy_alcohol = int(surveyData[10]['answerByUser'])

    # any healhcare insurance
    health_insurance = int(surveyData[11]['answerByUser'])

    # no doc because cost
    no_doc = int(surveyData[12]['answerByUser'])

    # general health
    gen_health = int(surveyData[13]['answerByUser'])

    # mental health
    mental_health = int(surveyData[14]['answerByUser'])

    # physical health
    physical_health = int(surveyData[15]['answerByUser'])

    # diff walking
    diff_walking = int(surveyData[16]['answerByUser'])

    # sex
    sex = int(surveyData[17]['answerByUser'])

    # age
    age = int(surveyData[18]['answerByUser'])

    # education
    education = int(surveyData[19]['answerByUser'])

    # income
    income = int(surveyData[20]['answerByUser'])

    data_predict = [high_bp, high_cholesterol, cholesterol_check, bmi, smoking, stroke, heart_disease, physical_activity, fruits, veggies, heavy_alcohol, health_insurance, no_doc, gen_health, mental_health, physical_health, diff_walking,
                    sex, age, education, income]

    result = model_rf.predict([data_predict])

    response = JsonResponse({'message': 'Hello, world. You\'re at the main index.',
                             'surveyData': surveyData, 'predict': json.dumps(result[0], cls=NpEncoder)})
    return response

def preprocessing_data(df):
    df_new = df
    # transform data
    df_new.Diabetes_012[df_new['Diabetes_012'] == 0] = 'No Diabetes'
    df_new.Diabetes_012[df_new['Diabetes_012'] == 1] = 'Prediabetes'
    df_new.Diabetes_012[df_new['Diabetes_012'] == 2] = 'Diabetes'

    df_new.HighBP[df_new['HighBP'] == 0] = 'No High'
    df_new.HighBP[df_new['HighBP'] == 1] = 'High BP'

    df_new.HighChol[df_new['HighChol'] == 0] = 'No High Cholesterol'
    df_new.HighChol[df_new['HighChol'] == 1] = 'High Cholesterol'

    df_new.CholCheck[df_new['CholCheck'] == 0] = 'No Cholesterol Check in 5 Years'
    df_new.CholCheck[df_new['CholCheck'] == 1] = 'Cholesterol Check in 5 Years'

    df_new.Smoker[df_new['Smoker'] == 0] = 'No'
    df_new.Smoker[df_new['Smoker'] == 1] = 'Yes'

    df_new.Stroke[df_new['Stroke'] == 0] = 'No'
    df_new.Stroke[df_new['Stroke'] == 1] = 'Yes'

    df_new.HeartDiseaseorAttack[df_new['HeartDiseaseorAttack'] == 0] = 'No'
    df_new.HeartDiseaseorAttack[df_new['HeartDiseaseorAttack'] == 1] = 'Yes'

    df_new.PhysActivity[df_new['PhysActivity'] == 0] = 'No'
    df_new.PhysActivity[df_new['PhysActivity'] == 1] = 'Yes'

    df_new.Fruits[df_new['Fruits'] == 0] = 'No'
    df_new.Fruits[df_new['Fruits'] == 1] = 'Yes'

    df_new.Veggies[df_new['Veggies'] == 0] = 'No'
    df_new.Veggies[df_new['Veggies'] == 1] = 'Yes'

    df_new.HvyAlcoholConsump[df_new['HvyAlcoholConsump'] == 0] = 'No'
    df_new.HvyAlcoholConsump[df_new['HvyAlcoholConsump'] == 1] = 'Yes'

    df_new.AnyHealthcare[df_new['AnyHealthcare'] == 0] = 'No'
    df_new.AnyHealthcare[df_new['AnyHealthcare'] == 1] = 'Yes'

    df_new.NoDocbcCost[df_new['NoDocbcCost'] == 0] = 'No'
    df_new.NoDocbcCost[df_new['NoDocbcCost'] == 1] = 'Yes'

    df_new.GenHlth[df_new['GenHlth'] == 1] = 'Excellent'
    df_new.GenHlth[df_new['GenHlth'] == 2] = 'Very Good'
    df_new.GenHlth[df_new['GenHlth'] == 3] = 'Good'
    df_new.GenHlth[df_new['GenHlth'] == 4] = 'Fair'
    df_new.GenHlth[df_new['GenHlth'] == 5] = 'Poor'

    df_new.DiffWalk[df_new['DiffWalk'] == 0] = 'No'
    df_new.DiffWalk[df_new['DiffWalk'] == 1] = 'Yes'

    df_new.Sex[df_new['Sex'] == 0] = 'Female'
    df_new.Sex[df_new['Sex'] == 1] = 'Male'

    df_new.Education[df_new['Education'] == 1] = 'Never Attended School'
    df_new.Education[df_new['Education'] == 2] = 'Elementary'
    df_new.Education[df_new['Education'] == 3] = 'Junior High School'
    df_new.Education[df_new['Education'] == 4] = 'Senior High School'
    df_new.Education[df_new['Education'] == 5] = 'Undergraduate Degree'
    df_new.Education[df_new['Education'] == 6] = 'Magister'

    df_new.Income[df_new['Income'] == 1] = 'Less Than $10,000'
    df_new.Income[df_new['Income'] == 2] = 'Less Than $10,000'
    df_new.Income[df_new['Income'] == 3] = 'Less Than $10,000'
    df_new.Income[df_new['Income'] == 4] = 'Less Than $10,000'
    df_new.Income[df_new['Income'] == 5] = 'Less Than $35,000'
    df_new.Income[df_new['Income'] == 6] = 'Less Than $35,000'
    df_new.Income[df_new['Income'] == 7] = 'Less Than $35,000'
    df_new.Income[df_new['Income'] == 8] = '$75,000 or More'
    return df_new

@api_view(['GET'])
def datasets_diabetes(request):
    data_frame = preprocessing_data(df_diabetes)
    diabetes_label = np.array(data_frame['Diabetes_012'].value_counts().index)
    diabetes_value = [x for x in data_frame['Diabetes_012'].value_counts()]

    smoking_label = np.array(data_frame['Smoker'].value_counts().index)
    smoking_value = [x for x in data_frame['Smoker'].value_counts()]

    cholesterol_label = np.array(data_frame['HighChol'].value_counts().index)
    cholesterol_value = [x for x in data_frame['HighChol'].value_counts()]

    blood_pressure_label = np.array(data_frame['HighBP'].value_counts().index)
    blood_pressure_value = [x for x in data_frame['HighBP'].value_counts()]

    return JsonResponse({
        "diabetes": {
            "label": diabetes_label.tolist(),
            "value": diabetes_value,
            "title": re.sub('([A-Z])', r' \1', "Heart Disease").title()
        },
        "smoking": {
            "label": smoking_label.tolist(),
            "value": smoking_value,
            "title": re.sub('([A-Z])', r' \1', "Smoking").title()
        },
        "cholesterol": {
            "label": cholesterol_label.tolist(),
            "value": cholesterol_value,
            "title": re.sub('([A-Z])', r' \1', "Cholesterol").title()
        },
        "blood_pressure": {
            "label": blood_pressure_label.tolist(),
            "value": blood_pressure_value,
            "title": re.sub('([A-Z])', r' \1', "Blood Pressure").title()
        }
    })

