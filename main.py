import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier as knn
from sklearn.metrics import confusion_matrix, classification_report

df = pd.read_csv("cancerData.csv")


X=df[['radius_mean' , 'perimeter_mean' , 'area_mean' , 'concave points_mean' , 'radius_worst' , 'perimeter_worst' , 'area_worst']]
y=df['diagnosis'].map({'M':1 , 'B':0})

X_train , X_test , y_train , y_test =train_test_split(X ,y , test_size=0.2 , random_state=42 )

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


clf = knn(n_neighbors=25)
clf.fit(X_train , y_train)

accuracy = clf.score(X_test , y_test)
print(f"Accuracy: {accuracy*100:.2f}%")


y_pred = clf.predict(X_test)
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Benign (0)', 'Malignant (1)']))
