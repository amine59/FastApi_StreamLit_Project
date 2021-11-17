import streamlit as st
import requests
#import json

class stream_lit:
    def dataselected(self):
        st.title("Amaris Demo FastApi")
        st.header("FastApi / StreamLit Dev and Deploy")
        self.sepal_length = st.text_input("sepal_length value cm")
        self.sepal_width = st.text_input("sepal_width value cm")
        self.petal_length = st.text_input("petal_length value cm")
        self.petal_width = st.text_input("petal_width value cm")

        self.data = {'sepal_length': self.sepal_length,
                'sepal_width': self.sepal_width,
                'petal_length': self.petal_length,
                'petal_width': self.petal_width}
        return self.data

    def front(self):

        self.type = st.sidebar.selectbox("Type d'algorithme", ("Classification", "Regression", "Clustering"))
        if self.type == "Classification":
            self.chosen_classifier = st.sidebar.selectbox("veuillez choisir l'algorithme",
                                                          ('Logistic Regression', 'Random Forest',
                                                           'Decision Tree Classifier', 'Support Vector Machines',
                                                           'Naive Bayes'))
        elif self.type == "Regression":
            self.chosen_classifier = st.sidebar.selectbox("veuillez choisir l'algorithme",
                                                          ('Random Forest', 'Linear Regression'))
        elif self.type == "Clustering":
            pass
        return self.type, self.chosen_classifier

    def predict(self, predict_btn):
        if self.type == "Classification":
            if self.chosen_classifier == 'Logistic Regression':
                self.req = requests.post("http://127.0.0.1:8000/predictLogisticRegression", json=self.data)
                self.prediction = self.req.text
                st.success(f"The prediction from Logistic Regression : {self.prediction}")

            elif self.chosen_classifier == 'Random Forest':
                self.req = requests.post("http://127.0.0.1:8000/predictRandomForest", timeout=8000)
                self.prediction = self.req.text
                st.success(f"The prediction from Random Forest: {self.prediction}")

            elif self.chosen_classifier == 'Decision Tree Classifier':
                self.req = requests.post("http://127.0.0.1:8000/DecisionTreeClassifier", json=self.data)
                self.prediction = self.req.text
                st.success(f"The prediction from Decision Tree: {self.prediction}")

            elif self.chosen_classifier == 'Support Vector Machines':
                self.req = requests.post("http://127.0.0.1:8000/SupportVectorMachines", json=self.data)
                self.prediction = self.req.text
                st.success(f"The prediction from SVM: {self.prediction}")

        if self.type == "Regression":
            if self.chosen_classifier == 'Linear Regression':
                st.write('yessssss')


if __name__ == '__main__':
    # by default it will run at 8501 port
    app = stream_lit()
    if len(app.dataselected()) > 1:
        app.front()
        predict_btn = st.sidebar.button('Predict')
    if predict_btn:
        st.sidebar.text("Progress:")
        my_bar = st.sidebar.progress(0)
        a = app.predict(predict_btn)
        for percent_complete in range(100):
            my_bar.progress(percent_complete + 1)