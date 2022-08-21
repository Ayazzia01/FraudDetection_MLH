import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

def load_data():
  return pd.read_csv('Fraud.csv')

def clean_data(df):
  large_amt = df[df['amount']>200000]
  large_amt = large_amt[large_amt['isFlaggedFraud'] == 0]
  large_amt = large_amt[large_amt['type'] == 'TRANSFER']
  df = pd.concat([df, large_amt]).drop_duplicates(keep=False)

  zero_amt = df[df['amount'] == 0]
  df = pd.concat([df, zero_amt]).drop_duplicates(keep=False)

  df1=pd.get_dummies(df, columns = ['type'], drop_first = True)
  
  return df1

def split_normalize_data(df):
  con_cols = ['amount', 'oldbalanceOrg', 'newbalanceOrig', 'step', 'oldbalanceDest', 'newbalanceDest']
  cust_cols = ['nameOrig', 'nameDest']

  y = df.pop('isFraud').astype('uint8').to_numpy()
  X = df.drop(columns=cust_cols, inplace=False)
  X['isFlaggedFraud'] = X['isFlaggedFraud'].astype('uint8')

  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, stratify=y, random_state=42)
  X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, stratify=y_train, random_state=42)

  std_scaler = StandardScaler()
  X_train[con_cols] = std_scaler.fit_transform(X_train[con_cols])
  X_test[con_cols] = std_scaler.transform(X_test[con_cols])
  X_val[con_cols] = std_scaler.transform(X_val[con_cols])

  data = {"train": {"X": X_train, "y": y_train},
          "test": {"X": X_test, "y": y_test},
          "val": {"X": X_val, "y": y_val},
  }

  return data

def oversample_data(data):
  oversample = SMOTE(sampling_strategy=0.1, random_state=42)
  X_train_oversample, y_train_oversample = oversample.fit_resample(data['train']['X'], data['train']['y'])

  data['train'] = {"X": X_train_oversample, "y": y_train_oversample}

  return data

def build_model():
  model = tf.keras.Sequential([
      tf.keras.layers.Dense(6, activation='relu', input_shape=(11,)),
      tf.keras.layers.Dropout(0.5),
      tf.keras.layers.Dense(1, activation='sigmoid'),
  ])

  return model

def train_model(model, data):
  metrics = [
      tf.keras.metrics.TruePositives(name='tp'),
      tf.keras.metrics.FalsePositives(name='fp'),
      tf.keras.metrics.TrueNegatives(name='tn'),
      tf.keras.metrics.FalseNegatives(name='fn'), 
      tf.keras.metrics.BinaryAccuracy(name='accuracy'),
      tf.keras.metrics.Precision(name='precision'),
      tf.keras.metrics.Recall(name='recall'),
      tf.keras.metrics.AUC(name='auc'),
      tf.keras.metrics.AUC(name='prc', curve='PR'), # precision-recall curve
  ]

  model.compile(
      optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3),
      loss = tf.keras.losses.BinaryCrossentropy(),
      metrics=metrics)
  
  callbacks = [
             tf.keras.callbacks.ModelCheckpoint('fraud_detection_model.pkl',
                                                monitor='val_prc',
                                                save_best_only=True,
                                                mode='max'),
             tf.keras.callbacks.EarlyStopping(monitor='val_prc',
                                              verbose=1,
                                              patience=10,
                                              mode='max',
                                              restore_best_weights=True)
  ]

  oversampled_class_weight = {0: (1 / len(data['train']['y'][data['train']['y']==0])) * (len(data['train']['y']) / 2.0),
                              1: (1 / len(data['train']['y'][data['train']['y']==1])) * (len(data['train']['y']) / 2.0)}

  model.fit(data['train']['X'], data['train']['y'],
            batch_size=2048,
            epochs=100,
            callbacks=callbacks,
            validation_data=(data['val']['X'], data['val']['y']),
            class_weight=oversampled_class_weight)

  return model

def get_model_metrics(model, data):
  metrics = {}
  test_metrics = model.evaluate(data['test']['X'], data['test']['X'], 2048)

  for metric_name, metric_value in zip(model.metrics_names, test_metrics):
    metrics[metric_name] = metric_value

  return metrics