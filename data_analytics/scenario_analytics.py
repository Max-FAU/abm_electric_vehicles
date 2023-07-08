from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
import pandas as pd

# Create a DataFrame with the given data
data = {
    'Scenario Number': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
    'Number of cars and customers': [25, 25, 25, 25, 50, 50, 50, 50, 150, 150, 150, 150, 300, 300, 300, 300],
    'Charging pattern': ['Uncontrolled', 'Uncontrolled', 'Off peak', 'Off peak', 'Uncontrolled', 'Uncontrolled',
                         'Off peak', 'Off peak', 'Uncontrolled', 'Uncontrolled', 'Off peak', 'Off peak',
                         'Uncontrolled', 'Uncontrolled', 'Off peak', 'Off peak'],
    'Plug-in behaviour': ['Home and work always', 'Home and work always', 'Home and work always',
                          'Home and work always', 'Home and work always', 'Home and work always',
                          'Home and work always', 'Home and work always', 'Home and work always',
                          'Home and work always', 'Home and work always', 'Home and work always',
                          'Home and work always', 'Home and work always', 'Home and work always',
                          'Home and work always'],
    'Charging Algorithm': ['False', 'True', 'False', 'True', 'False', 'True', 'False', 'True', 'False', 'True',
                           'False', 'True', 'False', 'True', 'False', 'True']
}

df = pd.DataFrame(data)


df_encoded = pd.get_dummies(df, columns=['Charging pattern', 'Plug-in behaviour', 'Charging Algorithm'])
X = df_encoded.drop('Scenario Number', axis=1)
y = df_encoded['Scenario Number']
model = DecisionTreeClassifier()
model.fit(X, y)

plt.figure(figsize=(20, 40))
plot_tree(model, feature_names=X.columns.tolist(), class_names=model.classes_.astype(str).tolist(), filled=True)
plt.tight_layout()
plt.show()