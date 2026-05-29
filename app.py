import streamlit as st
import pickle

# Load model and vectorizer
model = pickle.load(open("model/phishing_model.pkl", "rb"))
vectorizer = pickle.load(open("model/vectorizer.pkl", "rb"))

# Page title
st.title("🛡️ AI-Powered Phishing Email Detector")

st.write(
    "This system detects whether an email is Safe or Phishing using Machine Learning."
)

# User input
email_input = st.text_area("Enter Email Text")

# Predict button
if st.button("Detect Email"):

    if email_input.strip() != "":

        # Convert text
        email_vector = vectorizer.transform([email_input])

        # Prediction
        prediction = model.predict(email_vector)

        # Output
        if prediction[0] == "Phishing Email":
            st.error("⚠️ Warning! This is a Phishing Email.")
        else:
            st.success("✅ This is a Safe Email.")

    else:
        st.warning("Please enter email text.")