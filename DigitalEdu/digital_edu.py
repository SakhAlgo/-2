#создай здесь свой индивидуальный проект!
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

df = pd.read_csv('train.csv')

def set_age(row):
    if type(row['bdate']) == str:
        bdate = row['bdate'].split('.')
    else:
        bdate = ' '

    graduation = int(row['graduation'])
    if len(bdate) == 3 and len(bdate[2]) == 4:
        age = 2020 - int(bdate[2])
        return age
    elif 'Student' in row['education_status']:
        return 20
    elif graduation != 0 and graduation > 1950:
        return 2020 - (graduation - 23)

    

df['age'] = df.apply(set_age, axis=1)
# print(df['age'].value_counts())
df = df[df['age'].notna()]
df = df[['age', 'sex', 'education_status', 'relation', 'result', 'langs', 'occupation_type']]


def set_lang(lang):
    langs = lang.split(';')
    return len(langs)


df['langs'] = df['langs'].apply(set_lang)
# print(df['langs'].value_counts())

# 0 — None (нет образования);
# 1 — Undergraduate applicant (абитуриент);
# 2 — Student (Bachelor's) (студент бакалавриата);
# 3 — Alumnus (Bachelor's) (выпускник бакалавриата);
# 4 — Student (Master’s) (студент магистратуры);
# 5 — Alumnus (Master’s) (выпускник магистратуры);
# 6 — Candidate of Sciences (кандидат наук);
def set_ed_status(ed):
    if 'Undergraduate applicant' in ed:
        return 1
    elif "Student (Bachelor's)" in ed:
        return 2
    elif "Alumnus (Bachelor's)" in ed:
        return 3
    elif "Student (Master’s)" in ed:
        return 4
    elif "Alumnus (Master’s)" == ed:
        return 5
    elif "Candidate of Sciences" == ed:
        return 6
    return 0
        
df['education_status'] = df['education_status'].apply(set_ed_status)
print(df['education_status'].value_counts())
df[list(pd.get_dummies(df['occupation_type']).columns)] = pd.get_dummies(df['occupation_type'])
df = df[['age', 'sex', 'education_status', 'relation', 'result', 'langs']]
# df.info()

x = df.drop('result', axis=1)
y = df['result']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)
classifire = KNeighborsClassifier(n_neighbors=3)
classifire.fit(x_train, y_train)
y_pred = classifire.predict(x_test)
precent = accuracy_score(y_test, y_pred) * 100
print(precent)
print(confusion_matrix(y_test, y_pred))