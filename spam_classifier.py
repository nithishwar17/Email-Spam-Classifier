import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import string
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import numpy as np

# --- One-time NLTK Downloads ---
# Run these two lines once to download the necessary NLTK data
# nltk.download('stopwords')
# nltk.download('wordnet')


# --- STEP 1 & 2: Load and Structure the Data ---
# Replace with the actual path to your downloaded spam.csv file
file_path = 'e:/Cybernaut/Month 2/Email spam(mini project)/spam.csv' 

try:
    # Load the CSV, letting pandas handle the initial parsing
    df = pd.read_csv(file_path, encoding='latin-1')

    # The file has extra empty columns. We only need the first two.
    # Select all rows (:) and only the first two columns (:2)
    df = df.iloc[:, :2]

    # Rename the columns for clarity
    df.columns = ['label', 'message']
    
    print("Data loaded and structured successfully!")

except FileNotFoundError:
    print(f"Error: The file was not found at {file_path}")
    df = None
except Exception as e:
    print(f"An error occurred: {e}")
    df = None


# Proceed only if data was loaded successfully
if df is not None:

    # --- STEP 3: Data Preprocessing and Feature Extraction ---

    # Initialize the lemmatizer and stop words list
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))

    def clean_text(text):
        """A function to clean the raw text data."""
        # Check if the input is a string, otherwise return empty
        if not isinstance(text, str):
            return ""
        
        # 1. Convert to lowercase
        text = text.lower()
        # 2. Remove punctuation
        text = ''.join([char for char in text if char not in string.punctuation])
        # 3. Remove numbers
        text = ''.join([char for char in text if not char.isdigit()])
        # 4. Tokenize (split into words) and remove stop words
        words = [word for word in text.split() if word not in stop_words]
        # 5. Lemmatize words
        words = [lemmatizer.lemmatize(word) for word in words]
        
        return ' '.join(words)

    # Apply the cleaning function to the 'message' column
    print("\nCleaning text data...")
    df['cleaned_message'] = df['message'].apply(clean_text)
    print("Cleaning complete!")
    print("\nSample of cleaned data:")
    print(df.head())

    # Feature Extraction (Vectorization) using TF-IDF
    print("\nConverting text to numerical features using TF-IDF...")
    tfidf_vectorizer = TfidfVectorizer(max_features=3000) # Use top 3000 words as features
    X = tfidf_vectorizer.fit_transform(df['cleaned_message'])
    y = df['label'] # Our target variable
    print("Feature extraction complete!")
    print("Shape of feature matrix (X):", X.shape)

    
    # --- STEP 4: Model Training ---

    # Split the data into training and testing sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print(f"\nTraining set size: {X_train.shape[0]} messages")
    print(f"Testing set size: {X_test.shape[0]} messages")

    # Initialize and train the Naive Bayes classifier
    #print("\nTraining the Naive Bayes model...")
    #model = MultinomialNB()
    #model.fit(X_train, y_train)
    #print("Model training complete!")
    
    # Initialize and train the Logistic Regression model
    #print("\nTraining the Logistic Regression model...")
    #model = LogisticRegression(max_iter=1000)
    #model.fit(X_train, y_train)
    #print("Model training complete!")
    
    # Initialize and train the Support Vector Machine (SVM) model
    print("\nTraining the SVM model...")
    model = SVC()
    model.fit(X_train, y_train)
    print("Model training complete!")
    # --- STEP 5: Model Evaluation ---

    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Calculate accuracy and print a detailed report
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nModel Accuracy: {accuracy * 100:.2f}%")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # --- Visualize Top Spam Words (Corrected) ---
# Get the feature names (the words) from the TF-IDF vectorizer
# feature_names = np.array(tfidf_vectorizer.get_feature_names_out())
#
# Get the learned log-probabilities for each word for the 'spam' class
# We use index [1] because 'spam' is the second class.
# spam_word_scores = model.feature_log_prob_[1]
#
# Get the indices of the top 15 words with the highest scores
# top_word_indices = spam_word_scores.argsort()[-15:]
#
# --- Create and display the bar chart ---
# print("\nDisplaying a chart of the top 15 spam words...")
# plt.figure(figsize=(10, 7))
# plt.barh(feature_names[top_word_indices], spam_word_scores[top_word_indices], color='red')
# plt.title('Top 15 Words Indicating Spam')
# plt.xlabel('Log Probability Score')
# plt.ylabel('Words')
# plt.tight_layout()
# plt.show()

        # --- STEP 5: Model Evaluation ---
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nModel Accuracy: {accuracy * 100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # --- Visualize Top Spam Words (Naive Bayes Example) ---
    nb_model = MultinomialNB()
    nb_model.fit(X_train, y_train)

    feature_names = np.array(tfidf_vectorizer.get_feature_names_out())
    spam_word_scores = nb_model.feature_log_prob_[1]
    top_word_indices = spam_word_scores.argsort()[-15:]

    plt.figure(figsize=(10, 7))
    plt.barh(feature_names[top_word_indices], spam_word_scores[top_word_indices], color='red')
    plt.title('Top 15 Words Indicating Spam')
    plt.xlabel('Log Probability Score')
    plt.ylabel('Words')
    plt.tight_layout()
    plt.show()

    # --- STEP 6: Create a Predictive System ---
    def predict_message(message_text):
        ...


        # --- STEP 6: Create a Predictive System ---
    def predict_message(message_text):
        # Clean the input text using the same function from Step 3
        cleaned_text = clean_text(message_text)
        # Convert the cleaned text to a numerical vector using the same vectorizer
        vectorized_text = tfidf_vectorizer.transform([cleaned_text])
        # Use the trained SVM model to make a prediction
        prediction = model.predict(vectorized_text)
        return prediction[0]

    # --- Test it out with new messages! ---
    print("\n--- Testing the Predictive System ---")

    new_email_1 = "Congratulations! You've won a $1000 gift card. Call now to claim your prize."
    prediction_1 = predict_message(new_email_1)
    print(f"\nThe message: '{new_email_1}'")
    print(f"Is predicted to be: {prediction_1.upper()}")


    new_email_2 = "Hey, are we still on for the meeting tomorrow at 2pm? Let me know."
    prediction_2 = predict_message(new_email_2)
    print(f"\nThe message: '{new_email_2}'")
    print(f"Is predicted to be: {prediction_2.upper()}")

    # --- STEP 7: Interactive Prediction Loop ---
print("\n--- Spam Classifier is Ready ---")
print("Enter a message to check if it's spam or ham.")
print("Type 'quit' to exit.")

while True:
    # Ask the user for an email/message
    user_input = input("\nEnter your message here: ")
    
    # Check if the user wants to exit the loop
    if user_input.lower() == 'quit':
        print("Exiting classifier. Goodbye!")
        break
    
    # Use the predictive function to classify the message
    prediction = predict_message(user_input)
    
    # Print the result in a clear, user-friendly format
    print(f"--> Result: This message is predicted to be: {prediction.upper()}")