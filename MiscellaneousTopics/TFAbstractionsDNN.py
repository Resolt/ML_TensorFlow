from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split as TTS
from sklearn.preprocessing import MinMaxScaler as MMS

# GET THE DATA
wine_data = load_wine()
feat_data = wine_data['data']
labels = wine_data['target']

# SPLIT
X_train, X_test, y_train, y_test = TTS(feat_data, labels, test_size=0.3, random_state=64)

# SCALE
scaler = MMS()
scaled_x_train = scaler.fit_transform(X_train)
scaled_x_test = scaler.transform(X_test)

