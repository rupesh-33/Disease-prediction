from tkinter import *
import webbrowser as wb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_predict
from sklearn import linear_model, tree, ensemble


l1 =['itching','skin_rash','nodal_skin_eruptions','continuous_sneezing','shivering','chills','joint_pain',
'stomach_pain','acidity','ulcers_on_tongue','muscle_wasting','vomiting','burning_micturition','spotting_ urination',
'fatigue','weight_gain','anxiety','cold_hands_and_feets','mood_swings','weight_loss','restlessness','lethargy','patches_in_throat',
'irregular_sugar_level','cough','high_fever','sunken_eyes','breathlessness','sweating','dehydration','indigestion','headache','yellowish_skin',
'dark_urine','nausea','loss_of_appetite','pain_behind_the_eyes','back_pain','constipation','abdominal_pain','diarrhoea','mild_fever','yellow_urine',
'yellowing_of_eyes','acute_liver_failure','fluid_overload','swelling_of_stomach','swelled_lymph_nodes','malaise','blurred_and_distorted_vision',
'phlegm','throat_irritation','redness_of_eyes','sinus_pressure','runny_nose','congestion','chest_pain','weakness_in_limbs','fast_heart_rate',
'pain_during_bowel_movements','pain_in_anal_region','bloody_stool','irritation_in_anus','neck_pain','dizziness','cramps','bruising','obesity',
'swollen_legs','swollen_blood_vessels','puffy_face_and_eyes','enlarged_thyroid','brittle_nails','swollen_extremeties','excessive_hunger',
'extra_marital_contacts','drying_and_tingling_lips','slurred_speech','knee_pain','hip_joint_pain','muscle_weakness','stiff_neck','swelling_joints',
'movement_stiffness','spinning_movements','loss_of_balance','unsteadiness','weakness_of_one_body_side','loss_of_smell','bladder_discomfort',
'foul_smell_of urine','continuous_feel_of_urine','passage_of_gases','internal_itching','toxic_look_(typhos)','depression','irritability','muscle_pain',
'altered_sensorium','red_spots_over_body','belly_pain','abnormal_menstruation','dischromic _patches','watering_from_eyes','increased_appetite','polyuria',
'family_history','mucoid_sputum','rusty_sputum','lack_of_concentration','visual_disturbances','receiving_blood_transfusion','receiving_unsterile_injections',
'coma','stomach_bleeding','distention_of_abdomen','history_of_alcohol_consumption','fluid_overload','blood_in_sputum','prominent_veins_on_calf',
'palpitations','painful_walking','pus_filled_pimples','blackheads','scurring','skin_peeling','silver_like_dusting','small_dents_in_nails',
'inflammatory_nails','blister','red_sore_around_nose','yellow_crust_ooze']

disease=['Fungal infection','Allergy','GERD','Chronic cholestasis','Drug Reaction',
'Peptic ulcer diseae','AIDS','Diabetes','Gastroenteritis','Bronchial Asthma','Hypertension',
' Migraine','Cervical spondylosis',
'Paralysis (brain hemorrhage)','Jaundice','Malaria','Chicken pox','Dengue','Typhoid','hepatitis A',
'Hepatitis B','Hepatitis C','Hepatitis D','Hepatitis E','Alcoholic hepatitis','Tuberculosis',
'Common Cold','Pneumonia','Dimorphic hemmorhoids(piles)',
'Heart attack','Varicoseveins','Hypothyroidism','Hyperthyroidism','Hypoglycemia','Osteoarthristis',
'Arthritis','(vertigo) Paroymsal  Positional Vertigo','Acne','Urinary tract infection','Psoriasis',
'Impetigo']

l2=[]

for i in range(0,len(l1)):
    l2.append(0)


df=pd.read_csv("Prototype.csv")

df.replace({'prognosis':{'Fungal infection':0,'Allergy':1,'GERD':2,'Chronic cholestasis':3,'Drug Reaction':4,
'Peptic ulcer diseae':5,'AIDS':6,'Diabetes ':7,'Gastroenteritis':8,'Bronchial Asthma':9,'Hypertension ':10,
'Migraine':11,'Cervical spondylosis':12,
'Paralysis (brain hemorrhage)':13,'Jaundice':14,'Malaria':15,'Chicken pox':16,'Dengue':17,'Typhoid':18,'hepatitis A':19,
'Hepatitis B':20,'Hepatitis C':21,'Hepatitis D':22,'Hepatitis E':23,'Alcoholic hepatitis':24,'Tuberculosis':25,
'Common Cold':26,'Pneumonia':27,'Dimorphic hemmorhoids(piles)':28,'Heart attack':29,'Varicose veins':30,'Hypothyroidism':31,
'Hyperthyroidism':32,'Hypoglycemia':33,'Osteoarthristis':34,'Arthritis':35,
'(vertigo) Paroymsal  Positional Vertigo':36,'Acne':37,'Urinary tract infection':38,'Psoriasis':39,
'Impetigo':40}},inplace=True)

X= df[l1]

y = df[["prognosis"]]
np.ravel(y)

tr=pd.read_csv("Prototype-1.csv")

tr.replace({'prognosis':{'Fungal infection':0,'Allergy':1,'GERD':2,'Chronic cholestasis':3,'Drug Reaction':4,
'Peptic ulcer diseae':5,'AIDS':6,'Diabetes ':7,'Gastroenteritis':8,'Bronchial Asthma':9,'Hypertension ':10,
'Migraine':11,'Cervical spondylosis':12,
'Paralysis (brain hemorrhage)':13,'Jaundice':14,'Malaria':15,'Chicken pox':16,'Dengue':17,'Typhoid':18,'hepatitis A':19,
'Hepatitis B':20,'Hepatitis C':21,'Hepatitis D':22,'Hepatitis E':23,'Alcoholic hepatitis':24,'Tuberculosis':25,
'Common Cold':26,'Pneumonia':27,'Dimorphic hemmorhoids(piles)':28,'Heart attack':29,'Varicose veins':30,'Hypothyroidism':31,
'Hyperthyroidism':32,'Hypoglycemia':33,'Osteoarthristis':34,'Arthritis':35,
'(vertigo) Paroymsal  Positional Vertigo':36,'Acne':37,'Urinary tract infection':38,'Psoriasis':39,
'Impetigo':40}},inplace=True)

X_test= tr[l1]
y_test = tr[["prognosis"]]

np.ravel(y_test)

## reading dataset
d =['Drug Reaction','Malaria','Allergy','Hypothyroidism','Psoriasis','GERD','Chronic cholestasis','hepatitis A','Osteoarthristis',
'(vertigo) Paroymsal  Positional Vertigo','Hypoglycemia',
'Acne','Diabetes','Impetigo','Hypertension','Peptic ulcer diseae','Dimorphic hemmorhoids(piles)'
,'Common Cold','Chicken pox','Cervical spondylosis','Hyperthyroidism','Urinary tract infection',
'Varicose veins','AIDS','Paralysis (brain hemorrhage)','Typhoid','Hepatitis B'
,'Fungal infection','Hepatitis C','Migraine','Bronchial Asthma','Alcoholic hepatitis','Jaundice',
'Hepatitis E','Dengue','Hepatitis D','Heart attack','Pneumonia','Arthritis','Gastroenteritis','Tuberculosis']

d1 ={'Drug Reaction':14,'Malaria':29,'Allergy':4,'Hypothyroidism':26,'Psoriasis':35,'GERD':16,'Chronic cholestasis':9,'hepatitis A':40,
    'Osteoarthristis':31,'(vertigo) Paroymsal  Positional Vertigo':0,'Hypoglycemia':25,'Acne':2,'Diabetes':12,'Impetigo':27,'Hypertension':23,
    'Peptic ulcer diseae':33,'Dimorphic hemmorhoids(piles)':13,'Common Cold':10,'Chicken pox':8,'Cervical spondylosis':7,'Hyperthyroidism':24,
    'Urinary tract infection':38,'Varicose veins':39,'AIDS':1,'Paralysis (brain hemorrhage)':32,'Typhoid':37,'Hepatitis B':19,'Fungal infection':15,
    'Hepatitis C':20,'Migraine':30,'Bronchial Asthma':6,'Alcoholic hepatitis':3,'Jaundice':28,'Hepatitis E':22,'Dengue':11,'Hepatitis D':21,
    'Heart attack':18,'Pneumonia':34,'Arthritis':5,'Gastroenteritis':17,'Tuberculosis':36
    }


df1 = pd.read_csv("dp.csv")
df1.head(10)
pdf = df1.drop('Precaution_2',axis='columns')
ndf = pdf.dropna(axis=1)
(ndf.isna().sum())

def callback(url):
	wb.open_new_tab(url)

# GUI stuff..............................................................................
        
root = Tk()
root.configure(background='black')

Symptom1 = StringVar()
Symptom1.set("Select Here")

Symptom2 = StringVar()
Symptom2.set("Select Here")

Symptom3 = StringVar()
Symptom3.set("Select Here")

Symptom4 = StringVar()
Symptom4.set("Select Here")

Symptom5 = StringVar()
Symptom5.set("Select Here")

Name = StringVar()


img = PhotoImage(file="image.png",master=root)
mt = Label(root,image=img)
mt.place(x=0,y=0)


w2 = Label(root, justify=LEFT, text="Disease Predictor using Machine Learning", fg="white", bg="navy blue")
w2.config(font=("Times",10,"bold italic"))

w2.config(font=("Times",10,"bold italic"))
w2.grid(row=2, column=0, columnspan=2, padx=100)

NameLb = Label(root, text="Name of the Patient", fg="white", bg="navy blue")
NameLb.config(font=("Times",10,"bold italic"))
NameLb.grid(row=6, column=0, pady=15, sticky=W)

S1Lb = Label(root, text="Symptom 1", fg="white", bg="navy blue")
S1Lb.config(font=("Times",10,"bold italic"))
S1Lb.grid(row=7, column=0, pady=10, sticky=W)

S2Lb = Label(root, text="Symptom 2", fg="white", bg="navy blue")
S2Lb.config(font=("Times",10,"bold italic"))
S2Lb.grid(row=8, column=0, pady=10, sticky=W)

S3Lb = Label(root, text="Symptom 3", fg="white",bg="navy blue")
S3Lb.config(font=("Times",10,"bold italic"))
S3Lb.grid(row=9, column=0, pady=10, sticky=W)

S4Lb = Label(root, text="Symptom 4", fg="white", bg="navy blue")
S4Lb.config(font=("Times",10,"bold italic"))
S4Lb.grid(row=10, column=0, pady=10, sticky=W)

S5Lb = Label(root, text="Symptom 5", fg="white", bg="navy blue")
S5Lb.config(font=("Times",10,"bold italic"))
S5Lb.grid(row=11, column=0, pady=10, sticky=W)

destreeLb = Label(root, text="RandomForest", fg="Red", bg="navy blue")
destreeLb.config(font=("Times",10,"bold italic"))
destreeLb.grid(row=14, column=0, pady=10, sticky=W)

FD = Label(root,text ="Suggestion    ",fg="white",bg="navy blue")
FD.config(font=("Times",10,"bold italic"))
FD.grid(row=24, column=0, pady=10, sticky=W)

FD1 = Label(root,text ="Feedback    ",fg="white",bg="navy blue")
FD1.config(font=("Times",10,"bold italic"))
FD1.grid(row=26, column=0, pady=10, sticky=W)
FD1.bind("<Button-1>", lambda e:
callback("https://feedbackformsite.netlify.app/"))

OPTIONS = sorted(l1)

NameEn = Entry(root, textvariable=Name)
NameEn.grid(row=6, column=1)

S1 = OptionMenu(root, Symptom1,*OPTIONS)
S1.grid(row=7, column=1)

S2 = OptionMenu(root, Symptom2,*OPTIONS)
S2.grid(row=8, column=1)

S3 = OptionMenu(root, Symptom3,*OPTIONS)
S3.grid(row=9, column=1)

S4 = OptionMenu(root, Symptom4,*OPTIONS)
S4.grid(row=10, column=1)

S5 = OptionMenu(root, Symptom5,*OPTIONS)
S5.grid(row=11, column=1)

def randomforest():
    from sklearn.ensemble import RandomForestClassifier
    clf4 = RandomForestClassifier()
    clf4 = clf4.fit(X,np.ravel(y))

    # calculating accuracy 
    from sklearn.metrics import accuracy_score
    y_pred=clf4.predict(X_test)
    print(accuracy_score(y_test, y_pred))
    print(accuracy_score(y_test, y_pred,normalize=False))
    
    psymptoms = [Symptom1.get(),Symptom2.get(),Symptom3.get(),Symptom4.get(),Symptom5.get()]

    for k in range(0,len(l1)):
        for z in psymptoms:
            if(z==l1[k]):
                l2[k]=1

    inputtest = [l2]
    predict = clf4.predict(inputtest)
    predicted=predict[0]

    h='no'
    for a in range(0,len(disease)):
        if(predicted == a):
            h='yes'
            break

    if (h=='yes'):
        t2.delete("1.0", END)
        t2.insert(END, disease[a])
    else:
        t2.delete("1.0", END)
        t2.insert(END, "Not Found")

    

    XU = disease[a]
    i = ndf.drop('Precaution_1',axis='columns')
    t = ndf['Precaution_1']

    la_Disease = LabelEncoder()

    i['Disease_no'] = la_Disease.fit_transform(i['Disease'])
    i.head()

    n  = i.drop(['Disease'],axis='columns') 

    md = tree.DecisionTreeClassifier()
    md.fit(n,t)

    print(md.score(n,t))
    input = d1[disease[a]]
    ip_np = np.asarray(input)
    ip_rs = ip_np.reshape(1,-1)
    c = md.predict(ip_rs)
    print(c)
    t3.insert(INSERT,*c)

def Clear():
    t2.delete("1.0", END)
    t3.delete("1.0", END)
    Symptom1.set("Select Here")
    Symptom2.set("Select Here")
    Symptom3.set("Select Here")
    Symptom4.set("Select Here")
    Symptom5.set("Select Here")
    

rnf = Button(root, text="Predict", command=randomforest,bg="light pink",fg="navy blue")
rnf.config(font=("Times",15,"bold italic"))
rnf.grid(row=9, column=3,padx=10)

rnf = Button(root, text="clear", command=Clear,bg="light pink",fg="navy blue")
rnf.config(font=("Times",15,"bold italic"))
rnf.grid(row=9, column=5,padx=10)


t2 = Text(root, height=1, width=20,bg="navy blue",fg="red")
t2.config(font=("Times",15,"bold italic"))
t2.grid(row=14, column=1 , padx=10)

t3 = Text(root, height=1, width=20,bg="navy blue",fg="red")
t3.config(font=("Times",15,"bold italic"))
t3.grid(row=24, column=1, padx=10)


root.mainloop()