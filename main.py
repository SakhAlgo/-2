# Здесь должен быть твой код
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score


df = pd.read_csv('titanic.csv')
# print(df.info())
# print(df.groupby(by='Sex')['Survived'].mean())
# print(df.pivot_table(index='Survived', columns='Pclass', values='Age', aggfunc=('mean')))
# print(df.pivot_table(index='Survived', columns='Parch', values='Age', aggfunc=('mean')))
# print(df.pivot_table(index='Survived', columns='SibSp', values='Age', aggfunc=('mean')))

df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, inplace=True)

df['Embarked'].fillna('S', inplace=True)
print(df.info())
# print(df['Embarked'].value_counts())

# print(df.groupby(by='Pclass')['Age'].median())
def get_age(row):
    if pd.isnull(row['Age']):
        if row['Pclass'] == 1:
            return 37
        elif row['Pclass'] == 2:
            return 29
        return 24
    return row['Age']
    
df['Age'] = df.apply(get_age, axis = 1)
# print(pd.get_dummies(df['Embarked']))
df[list(pd.get_dummies(df['Embarked']).columns)] = pd.get_dummies(df['Embarked'])
df.drop('Embarked', axis=1, inplace=True)

def set_sex(sex):
    if set == 'female':
        return 0
    return 1


df['Sex'] = df['Sex'].apply(set_sex)

def get_alone(row):
    if row['SibSp'] + row['Parch'] == 0:
        return True
    return False


df['Alone'] = df.apply(get_alone, axis=1)
# print(df.pivot_table(index='Survived', columns='Alone', values='Sex', aggfunc='count'))
# print(df.info())
x = df.drop('Survived', axis=1)
y = df['Survived']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)
sc = StandardScaler()

x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)
classifire = KNeighborsClassifier(n_neighbors=3)
classifire.fit(x_train, y_train)
y_pred = classifire.predict(x_test)
precent = accuracy_score(y_test, y_pred) * 100
print(precent)
print(confusion_matrix(y_test, y_pred))git remote add origin https://github.com/SakhAlgo/ML.git