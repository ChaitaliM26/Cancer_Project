from flask import Flask,render_template,request,redirect
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler

scaler=StandardScaler()

model=pickle.load(open("model.pkl","rb"))

app=Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict",methods=["POST"])
def Breast_Cancer_Prediction():
    radius_mean=float(request.form.get("radius_mean"))
    texture_mean=float(request.form.get("texture_mean"))
    perimeter_mean=float(request.form.get("perimeter_mean"))
    area_mean=float(request.form.get("area_mean"))
    smoothness_mean=float(request.form.get("smoothness_mean"))
    compactness_mean=float(request.form.get("compactness_mean"))
    concavity_mean=float(request.form.get("concavity_mean"))
    concave_points_mean=float(request.form.get("concave points_mean"))
    symmetry_mean=float(request.form.get("symmetry_mean"))
    fractal_dimension_mean=float(request.form.get("fractal_dimension_mean"))
    radius_se=float(request.form.get("radius_se"))
    texture_se=float(request.form.get("texture_se"))
    perimeter_se=float(request.form.get("perimeter_se"))
    area_se=float(request.form.get("area_se"))
    smoothness_se=float(request.form.get("smoothness_se"))
    compactness_se=float(request.form.get("compactness_se"))
    concavity_se=float(request.form.get("concavity_se"))
    concave_points_se=float(request.form.get("concave points_se"))
    symmetry_se=float(request.form.get("symmetry_se"))
    fractal_dimension_se=float(request.form.get("fractal_dimension_se"))
    radius_worst=float(request.form.get("radius_worst"))
    texture_worst=float(request.form.get("texture_worst"))
    perimeter_worst=float(request.form.get("perimeter_worst"))
    area_worst=float(request.form.get("area_worst"))
    smoothness_worst=float(request.form.get("smoothness_worst"))
    compactness_worst=float(request.form.get("compactness_worst"))
    concavity_worst=float(request.form.get("concavity_worst"))
    concave_points_worst=float(request.form.get("concave points_worst"))
    symmetry_worst=float(request.form.get("symmetry_worst"))
    fractal_dimension_worst=float(request.form.get("fractal_dimension_worst"))

    x_test=np.array([['radius_mean', 'texture_mean', 'perimeter_mean',
       'area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean',
       'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean',
       'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
       'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se',
       'fractal_dimension_se', 'radius_worst', 'texture_worst',
       'perimeter_worst', 'area_worst', 'smoothness_worst',
       'compactness_worst', 'concavity_worst', 'concave points_worst',
       'symmetry_worst', 'fractal_dimension_worst']])
    x_test_scaled=scaler.transform(x_test)

    result=model.predict(x_test_scaled)
    
    # return render_template('index.html', chance_of_admit='student has a probability to get admission  {}'.format(result))

 
    if result==1:
       return "<h1 style='color:green'>M</h1>"
    else:
       return "<h1 style='color:red'>B</h1>"
    
    
# @app.route("/predict",methods=["GET"])
# def predict_placement():
#     cgpa=float(request.args.get("cgpa"))
#     iq=float(request.args.get("iq"))
#     profile_score=float(request.args.get("profile_score"))
    
    
#     result=model.predict(np.array([[cgpa,iq,profile_score]]))
    
#     if result[0]==1:
#         return "<h1 style='color:green'>PLACED</h1>"
#     else:
#         return "<h1 style='color:red'>NOT PLACED</h1>"   

if __name__=='__main__':
    app.run(host="127.0.0.1",port=5000,debug=True)