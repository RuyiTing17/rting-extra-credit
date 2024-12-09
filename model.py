import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix
import xgboost as xgb

def preprocessing(df, label_encoders=None, scaler=None, is_train=True):
    df['trans_datetime'] = pd.to_datetime(df['unix_time'], unit='s')
    df['hour'] = df['trans_datetime'].dt.hour
    df['day_of_week'] = df['trans_datetime'].dt.dayofweek
    df['day_of_year'] = df['trans_datetime'].dt.dayofyear
    df['weekend'] = df['trans_datetime'].dt.weekday >= 5
    df['dob'] = pd.to_datetime(df['dob'])
    df['age'] = (df['trans_datetime'] - df['dob']).dt.days // 365
    df['transactions_per_user'] = df.groupby('id')['id'].transform('count')
    df['avg_amt_per_user'] = df.groupby('id')['amt'].transform('mean')
    df['amt_deviation'] = df['amt'] - df['avg_amt_per_user']
    df['amt_transactions'] = df['amt'] * df['transactions_per_user']
    df.drop(columns=['unix_time', 'trans_datetime'], inplace=True, errors='ignore')

    numerical = ['amt', 'avg_amt_per_user', 'amt_deviation', 'amt_transactions']

    if is_train:
        scaler = MinMaxScaler()
        df[numerical] = scaler.fit_transform(df[numerical])
    else:
        df[numerical] = scaler.transform(df[numerical])


    categorical = ['category', 'gender', 'state', 'job', 'merchant']

    if is_train:
        label_encoders = {col: LabelEncoder() for col in categorical}
        for col, le in label_encoders.items():
            df[col] = le.fit_transform(df[col])
    else:
        for col, le in label_encoders.items():
            df[col] = le.transform(df[col])

    
    select = [
        'category', 'amt', 'hour', 'day_of_week', 'gender', 'state',
        'city_pop', 'job', 'day_of_year', 'weekend', 'transactions_per_user',
        'age', 'avg_amt_per_user', 'amt_deviation', 'amt_transactions'
    ]

    return df[select], label_encoders, scaler

# Training data
train_data = pd.read_csv('train.csv')
X_train, label_encoders, scaler = preprocessing(train_data, is_train=True)
y_train = train_data['is_fraud']

xgb_model = xgb.XGBClassifier(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    objective='binary:logistic',
    tree_method='hist',
    predictor='auto', 
    random_state=42
)

X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
    X_train, y_train, test_size=0.2, stratify=y_train, random_state=42
)

# Train
xgb_model.fit(X_train_split, y_train_split)

# Validate
y_val_pred = xgb_model.predict(X_val_split)
accuracy = accuracy_score(y_val_split, y_val_pred)
f1 = f1_score(y_val_split, y_val_pred)
matrix = confusion_matrix(y_val_split, y_val_pred)
report = classification_report(y_val_split, y_val_pred)

print("Validation Accuracy:", accuracy)
print("F1 Score:", f1)
print("Confusion Matrix:\n", matrix)
print("Classification Report:\n", report)

# Load test data
test_data = pd.read_csv('test.csv')
X_test, _, _ = preprocessing(test_data, label_encoders=label_encoders, scaler=scaler, is_train=False)

# Predict on test data
test_data['is_fraud'] = xgb_model.predict(X_test)

test_data[['id', 'is_fraud']].to_csv('submission.csv', index=False)

print(f"saved.")
