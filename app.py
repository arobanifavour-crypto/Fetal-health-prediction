import os
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

@st.cache_resource
def load_data(path='fetalhealth.csv'):
	return pd.read_csv(path)

@st.cache_resource
def load_or_train_model(df):
	if os.path.exists('model.pkl'):
		model = joblib.load('model.pkl')
		return model

	# Train a quick model if none exists
	X = df.drop('fetal_health', axis=1)
	y = df['fetal_health']
	pipeline = Pipeline([
		('scaler', StandardScaler()),
		('clf', RandomForestClassifier(n_estimators=100, random_state=42))
	])
	pipeline.fit(X, y)
	joblib.dump(pipeline, 'model.pkl')
	return pipeline

def main():
	st.title('Fetal Health Classification')
	st.markdown('Upload data or enter values to predict fetal health (1=Normal, 2=Suspect, 3=Pathologic)')

	df = load_data()
	model = load_or_train_model(df)

	st.sidebar.header('Prediction inputs')
	feature_names = list(df.columns[:-1])
	defaults = df.median()
	user_input = {}
	for feat in feature_names:
		# use median as default; allow float step
		val = st.sidebar.number_input(feat, value=float(defaults[feat]), format="%f")
		user_input[feat] = val

	if st.sidebar.button('Predict'):
		input_df = pd.DataFrame([user_input])
		pred = model.predict(input_df)[0]
		proba = None
		try:
			proba = model.predict_proba(input_df)[0]
		except Exception:
			pass

		label_map = {1: 'Normal', 2: 'Suspect', 3: 'Pathologic'}
		st.subheader('Prediction')
		st.write('Class:', int(pred), '-', label_map.get(int(pred), 'Unknown'))
		if proba is not None:
			st.write('Probabilities:')
			proba_ser = pd.Series(proba, index=[label_map.get(i+1, str(i+1)) for i in range(len(proba))])
			st.dataframe(proba_ser.to_frame('probability'))

	st.sidebar.markdown('---')
	if st.sidebar.checkbox('Show raw data'):
		st.subheader('Dataset sample')
		st.dataframe(df.head())

	st.subheader('Class distribution')
	st.bar_chart(df['fetal_health'].value_counts())

	st.subheader('Feature preview')
	st.dataframe(df[feature_names].describe().T)

if __name__ == '__main__':
	main()
