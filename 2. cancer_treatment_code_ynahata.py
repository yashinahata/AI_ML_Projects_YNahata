"""
Automated Classification of Genetic Mutations for Personalized Cancer Treatment using Machine Learning
Author: Yashi Nahata
Dataset: Memorial Sloan Kettering Cancer Center (MSKCC)
Model deployed: Logistic regression
References: Iker Huerga, Wendy Kan. (2017). Personalized Medicine: Redefining Cancer Treatment.
            https://kaggle.com/competitions/msk-redefining-cancer-treatment
"""

# 1. Importing relevant libraries
import pandas as pd
import numpy as py
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import log_loss
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import normalize
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import seaborn as sns

# 2. Reading and exploring the data
training_variants = pd.read_csv('C:/Users/yasha/project_2/data/training_variants')
training_text = pd.read_csv('C:/Users/yasha/project_2/data/training_text', sep="\|\|",engine="python",names=["ID","TEXT"],skiprows=1)
test_variants = pd.read_csv('C:/Users/yasha/project_2/data/test_variants')
test_text = pd.read_csv('C:/Users/yasha/project_2/data/test_text', sep="\|\|",engine="python",names=["ID","TEXT"],skiprows=1)
training_variants.head()
training_text.head()
training_variants.shape
training_text.shape
test_variants.shape
test_text.shape

# 3. Checking and cleaning the data
train_data = pd.merge(training_text, training_variants, on = "ID", how = "left")
train_data.isnull().values.any()
train_data.loc[train_data['TEXT'].isnull(), 'TEXT'] = train_data['Gene'] + ' ' + train_data['Variation']
train_data.isnull().values.any()
train_data.head()
test_data = pd.merge(test_text, test_variants, on = "ID", how = "left")
test_data.isnull().values.any()
test_data.loc[test_data['TEXT'].isnull(), 'TEXT'] = test_data['Gene'] + ' ' + test_data['Variation']
test_data.isnull().values.any()
test_data.head()

# 4. Splitting the data into training, validation and testing datasets
y_true = (train_data['Class'])
del train_data['Class']
x_train, x_rem, y_train, y_rem = train_test_split(train_data, y_true, stratify = y_true, test_size = 0.2)
x_val, x_test, y_val, y_test = train_test_split(x_rem, y_rem, stratify = y_rem, test_size = 0.5)

# 5. Defining the function to analyze model prediction correctness
def evaluate_model (model, test_features, y_truth, datatype = ''):
    pred = model.predict(test_features)
    cm = confusion_matrix(y_truth, pred)
    sns.heatmap(cm, annot=True, fmt='g')
    plt.ylabel('Prediction', fontsize=13)
    plt.xlabel('Actual', fontsize=13)
    plt.title('Confusion Matrix', fontsize = 17)
    plt.show()
    y_truth.shape
    pred.shape
    pred_prob = model.predict_proba(test_features)
    eval = log_loss(y_truth, pred_prob)
    print("Log loss for " + datatype + " data")
    print(eval)
    print(cm)
    print('************************************************')
    
# 6. Preparing the datasets for modeling - transforming categorical variables, extracting relevant text variables
train_gene_feature_onehotcoding = pd.get_dummies(x_train["Gene"], drop_first = True)
val_gene_feature_onehotcoding = pd.get_dummies(x_val["Gene"], drop_first = True)
val_gene_feature_onehotcoding = val_gene_feature_onehotcoding.reindex(columns = train_gene_feature_onehotcoding.columns, fill_value = 0)
test_gene_feature_onehotcoding = pd.get_dummies(x_test["Gene"], drop_first = True)
test_gene_feature_onehotcoding = test_gene_feature_onehotcoding.reindex(columns = train_gene_feature_onehotcoding.columns, fill_value = 0)
train_var_feature_onehotcoding = pd.get_dummies(x_train["Variation"], drop_first = True)
val_var_feature_onehotcoding = pd.get_dummies(x_val["Variation"], drop_first = True)
val_var_feature_onehotcoding = val_var_feature_onehotcoding.reindex(columns = train_var_feature_onehotcoding.columns, fill_value = 0)
test_var_feature_onehotcoding = pd.get_dummies(x_test["Variation"], drop_first = True)
test_var_feature_onehotcoding = test_var_feature_onehotcoding.reindex(columns = train_var_feature_onehotcoding.columns, fill_value = 0)
text_vectorizer = TfidfVectorizer(max_df=0.7, stop_words="english")
train_text_feature_onehotcoding = text_vectorizer.fit_transform(x_train['TEXT'])
val_text_feature_onehotcoding = text_vectorizer.fit_transform(x_val['TEXT'])
test_text_feature_onehotcoding = text_vectorizer.fit_transform(x_test['TEXT'])
train_text_feature_onehotcoding = normalize(train_text_feature_onehotcoding, axis =0)
val_text_feature_onehotcoding = normalize(val_text_feature_onehotcoding, axis =0)
test_text_feature_onehotcoding = normalize(test_text_feature_onehotcoding, axis =0)
train_text_feature_onehotcoding = pd.DataFrame(train_text_feature_onehotcoding.toarray())
val_text_feature_onehotcoding = pd.DataFrame(val_text_feature_onehotcoding.toarray())
test_text_feature_onehotcoding = pd.DataFrame(test_text_feature_onehotcoding.toarray())
gene_variation_train = pd.concat([ train_var_feature_onehotcoding,train_gene_feature_onehotcoding], axis = 1)
gene_variation_val = pd.concat([ val_var_feature_onehotcoding, val_gene_feature_onehotcoding], axis = 1)
gene_variation_test = pd.concat([test_var_feature_onehotcoding , test_gene_feature_onehotcoding], axis = 1)
gene_variation_train.reset_index(drop=True, inplace = True)
gene_variation_test.reset_index(drop=True, inplace = True)
gene_variation_val.reset_index(drop=True, inplace = True)
text_gene_variation_train = pd.concat([train_text_feature_onehotcoding, gene_variation_train], axis = 1)
text_gene_variation_val = pd.concat([val_text_feature_onehotcoding, gene_variation_val], axis = 1)
text_gene_variation_test = pd.concat([test_text_feature_onehotcoding,gene_variation_test], axis = 1)
text_gene_variation_train.columns = text_gene_variation_train.columns.astype(str)
text_gene_variation_test.columns = text_gene_variation_test.columns.astype(str)
text_gene_variation_val.columns = text_gene_variation_val.columns.astype(str)
not_existing_cols = [c for c in text_gene_variation_train.columns.tolist() if c not in text_gene_variation_val]
text_gene_variation_val = text_gene_variation_val.reindex(text_gene_variation_val.columns.tolist() + not_existing_cols, axis=1)
text_gene_variation_val.fillna(0,inplace=True)
text_gene_variation_val = text_gene_variation_val[text_gene_variation_train.columns.tolist()]
not_existing_cols1 = [a for a in text_gene_variation_train.columns.tolist() if a not in text_gene_variation_test]
text_gene_variation_test = text_gene_variation_test.reindex(text_gene_variation_test.columns.tolist() + not_existing_cols1, axis=1)
text_gene_variation_test.fillna(0,inplace=True)
text_gene_variation_test = text_gene_variation_test[text_gene_variation_train.columns.tolist()]

# 7. Running and evaluating the model in training, validation and testing datasets
loj= LogisticRegression(random_state=0)
log_model = loj.fit(text_gene_variation_train, y_train)
evaluate_model(log_model, text_gene_variation_train, y_train, 'training')
evaluate_model(log_model, text_gene_variation_val, y_val, 'validation')
evaluate_model(log_model, text_gene_variation_test, y_test, 'testing')

# 8. Preparing the new (unseen) data for modeling
test_variation_onehotcoding = pd.get_dummies(test_data["Variation"], drop_first = True)
test_variation_onehotcoding = test_variation_onehotcoding.reindex(columns = train_var_feature_onehotcoding.columns, fill_value=0)
test_gene_onehotcoding = pd.get_dummies(test_data['Gene'], drop_first = True)
test_gene_onehotcoding = test_gene_onehotcoding.reindex(columns = train_gene_feature_onehotcoding.columns, fill_value=0)
test_text_onehotcoding = text_vectorizer.transform(test_data['TEXT'])
test_text_onehotcoding = normalize(test_text_onehotcoding, axis=0)
test_text_onehotcoding = pd.DataFrame(test_text_onehotcoding.toarray()) 
gene_variation =pd.concat([test_variation_onehotcoding, test_gene_onehotcoding], axis=1)
gene_variation = gene_variation.reset_index(drop= True, inplace=True)
gene_variation_text = pd.concat([test_text_onehotcoding, gene_variation], axis=1)
gene_variation_text.columns = gene_variation_text.columns.astype(str)
not_existing_cols2 = [b for b in text_gene_variation_train.columns.tolist() if b not in gene_variation_text]
gene_variation_text = gene_variation_text.reindex(gene_variation_text.columns.tolist() + not_existing_cols2, axis=1)
gene_variation_text.fillna(0,inplace=True)
gene_variation_text = gene_variation_text[text_gene_variation_train.columns.tolist()]

# 9. Running the model on unseen data to predict output classes
pred_prob = log_model.predict(gene_variation_text)
print("Predicted classes for first 20 entries:")
print(pred_prob[:20])
