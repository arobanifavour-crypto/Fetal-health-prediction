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
	st.markdown('Upload the fetal health dataset (CSV) or enter values to predict fetal health (1=Normal, 2=Suspect, 3=Pathologic)')

	# Add file uploader for the dataset
	uploaded_file = st.file_uploader("Upload fetalhealth.csv", type="csv")
	
	if uploaded_file is not None:
		df = pd.read_csv(uploaded_file)
		st.success("Dataset uploaded successfully!")
	else:
		st.warning("Please upload the fetalhealth.csv file to proceed. Using default values for now.")
		# Fallback: Create a dummy DataFrame with expected columns for defaults (adjust as needed)
		# This assumes standard fetal health features; replace with actual defaults if known
		feature_names = [
			'baseline value', 'accelerations', 'fetal_movement', 'uterine_contractions',
			'light_decelerations', 'severe_decelerations', 'prolongued_decelerations',
			'abnormal_short_term_variability', 'mean_value_of_short_term_variability',
			'percentage_of_time_with_abnormal_long_term_variability',
			'mean_value_of_long_term_variability', 'histogram_width', 'histogram_min',
			'histogram_max', 'histogram_number_of_peaks', 'histogram_number_of_zeroes',
			'histogram_mode', 'histogram_mean', 'histogram_median', 'histogram_variance',
			'histogram_tendency'
		]
		df = pd.DataFrame({feat: [0.0] for feat in feature_names + ['fetal_health']})  # Dummy row
		defaults = pd.Series({feat: 0.0 for feat in feature_names})  # Default to 0 if no data

	# Load or train model (now uses the uploaded/fallback df)
	model = load_or_train_model(df)

	st.sidebar.header('Prediction inputs')
	if 'defaults' not in locals():
		defaults = df.median()  # Use medians from uploaded data
	user_input = {}
	for feat in feature_names:
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
	if st.sidebar.checkbox('Show raw data') and uploaded_file is not None:
		st.subheader('Dataset sample')
		st.dataframe(df.head())

	if uploaded_file is not None:
		st.subheader('Class distribution')
		st.bar_chart(df['fetal_health'].value_counts())

		st.subheader('Feature preview')
		st.dataframe(df[feature_names].describe().T)

if __name__ == '__main__':
	main()
