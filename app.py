
import streamlit as st
import pandas as pd
import joblib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

# Title of the app
st.title("DNA Sequence Classification App")

# Load preprocessed data if already saved
@st.cache_data
def load_data():
    human_data = pd.read_csv('human_data_processed.csv')
    chimp_data = pd.read_csv('chimp_data_processed.csv')
    dog_data = pd.read_csv('dog_data_processed.csv')
    return human_data, chimp_data, dog_data

human_data, chimp_data, dog_data = load_data()

# Load the saved model and vectorizer
@st.cache_resource
def load_model_and_vectorizer():
    cv = joblib.load('count_vectorizer.pkl')
    classifier = joblib.load('naive_bayes_model.pkl')
    return cv, classifier

cv, classifier = load_model_and_vectorizer()

# Show processed data if checkbox is selected
if st.checkbox('Show processed human data'):
    st.subheader("Processed Human Data")
    st.write(human_data.head())

# Train/Test split and model training option
if st.checkbox('Retrain the Model'):
    st.write("Retraining the model...")
    
    # Convert sequences to k-mer words and train-test split
    X = cv.fit_transform(human_data['words'])
    y_data = human_data.iloc[:, 0].values
    X_train, X_test, y_train, y_test = train_test_split(X, y_data, test_size=0.20, random_state=42)
    
    # Train the model
    classifier = MultinomialNB(alpha=0.1)
    classifier.fit(X_train, y_train)
    
    # Save the retrained model
    joblib.dump(classifier, 'naive_bayes_model.pkl')
    st.success("Model retrained and saved successfully!")
    
    # Evaluate the model
    y_pred = classifier.predict(X_test)
    accuracy, precision, recall, f1 = accuracy_score(y_test, y_pred), precision_score(y_test, y_pred, average='weighted'), recall_score(y_test, y_pred, average='weighted'), f1_score(y_test, y_pred, average='weighted')
    st.write(f"Accuracy: {accuracy:.3f}")
    st.write(f"Precision: {precision:.3f}")
    st.write(f"Recall: {recall:.3f}")
    st.write(f"F1 Score: {f1:.3f}")

# Input DNA sequence for prediction
st.subheader("Predict Gene Function from DNA Sequence")
user_input = st.text_area("Enter DNA sequence here:", height=150)

if st.button('Predict'):
    if len(user_input) >= 6:
        user_kmers = [' '.join([user_input[i:i+6] for i in range(len(user_input) - 6 + 1)])]
        user_vector = cv.transform(user_kmers)
        prediction = classifier.predict(user_vector)
        st.write(f'Predicted class: **{prediction[0]}**')
    else:
        st.error("DNA sequence must be at least 6 bases long.")

# Optionally display chimp and dog data
if st.checkbox('Show processed chimp data'):
    st.subheader("Processed Chimp Data")
    st.write(chimp_data.head())

if st.checkbox('Show processed dog data'):
    st.subheader("Processed Dog Data")
    st.write(dog_data.head())

# Footer
st.write("Built with ❤️ using Streamlit")
