# 1. Library imports
import uvicorn
from fastapi import FastAPI
from model import IrisModelRandomForest, IrisModelLogisticRegression, DecisionTreeClassifier,SupportVectorMachines, IrisSpecies

app = FastAPI()
model = IrisModelRandomForest()
model2 = IrisModelLogisticRegression()
model3 = DecisionTreeClassifier()
model4 = SupportVectorMachines()

@app.post('/predictRandomForest')
def predict_RandomForest(iris: IrisSpecies):
    dataframe = iris.dict()
    prediction, probability = model.predict_RandomForest(
        dataframe['sepal_length'], dataframe['sepal_width'], dataframe['petal_length'], dataframe['petal_width']
    )
    return {
        'prediction': prediction,
        'probability': probability
    }

@app.post('/predictLogisticRegression')
def predict_LogisticRegression(iris: IrisSpecies):
    dataframe = iris.dict()
    prediction, probability = model2.predict_LogisticRegression(
        #dataframe['sepal_length'], dataframe['sepal_width'], dataframe['petal_length'], dataframe['petal_width']
        iris.dict()
    )
    return {
        'prediction': prediction,
        'probability': probability
    }

@app.post('/DecisionTreeClassifier')
def predict_DecisionTree(iris: IrisSpecies):
    dataframe = iris.dict()
    prediction, probability = model3.predict_DecisionTree(
        #dataframe['sepal_length'], dataframe['sepal_width'], dataframe['petal_length'], dataframe['petal_width']
        iris.dict()
    )
    return {
        'prediction': prediction,
        'probability': probability
    }

@app.post('/SupportVectorMachines')
def predict_SupportVectorMachines(iris: IrisSpecies):
    dataframe = iris.dict()
    prediction, probability = model4.predict_SupportVectorMachines(
        #dataframe['sepal_length'], dataframe['sepal_width'], dataframe['petal_length'], dataframe['petal_width']
        iris.dict()
    )
    return {
        'prediction': prediction,
        'probability': probability
    }


# 4. Run the API with uvicorn
#    Will run on http://127.0.0.1:8000
if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)