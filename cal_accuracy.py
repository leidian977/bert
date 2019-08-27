import pandas as pd
from sklearn.metrics import accuracy_score

df_test = pd.read_csv("test.csv")
df_pred = pd.read_csv("dataset/output/test_results.tsv", sep='\t', header=-1)

df_pred['label'] = df_pred.idxmax(axis=1)
df_pred.label.replace({0:'unrelated', 1:'agreed', 2:'disagreed'}, inplace=True)

acc = accuracy_score(df_test['label'], df_pred['label'])
print("Accuracy:{}".format(acc))