from flask import Flask, render_template, request
app = Flask(__name__)

import pickle

file = open('model.pkl','rb')
clf = pickle.load(file)
file.close()

@app.route('/', methods=["GET", "POST"])
def corona_test_kit():
    if request.method == "POST":
        myDict = request.form
        MedicalConditions = int(myDict['MedicalConditions'])
        RunningNose = int(myDict['RunningNose'])
        Temperature = int(myDict['Temperature'])
        Gender = int(myDict['Gender'])
        Age = int(myDict['Age'])
        Myalgia = int(myDict['Myalgia'])
        Vomiting = int(myDict['Vomiting'])
        Fatigue = int(myDict['Fatigue'])
        Throatache = int(myDict['Throatache'])     
        Cough = int(myDict['Cough'])
        Headache = int(myDict['Headache'])        

        if Temperature > 101:
            Fever = 1
        else:
            Fever = 0

        #%% code for inference
        inputFeatures = [Age,Gender,Temperature,MedicalConditions,RunningNose,Cough,Myalgia,Headache,Throatache,Fever,Fatigue,Vomiting]
        InfectionProbability = clf.predict_proba([inputFeatures])[0][1]
        InfectionProbability = InfectionProbability * 100

        if ( InfectionProbability >= 50 and InfectionProbability < 75 ):
            result = "Considerable chances of having COVID. Please consult with a doctor."
        elif ( InfectionProbability >= 75 ):
            result = "You may have common flu. Please consult a doctor."
        else:
            result = "You are safe. Stay home and save the world."         
            
        return render_template('show.html',InfectionProbability=InfectionProbability, result=result)
        
        
    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)
