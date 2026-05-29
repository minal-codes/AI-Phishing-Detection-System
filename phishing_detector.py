import pickle

# Load saved model and vectorizer
model = pickle.load(open("model/phishing_model.pkl", "rb"))
vectorizer = pickle.load(open("model/vectorizer.pkl", "rb"))

# Sample email
email = input("Enter email text: ")

# Convert text
email_vector = vectorizer.transform([email])

# Predict
prediction = model.predict(email_vector)

# Output
print("\nPrediction:", prediction[0])