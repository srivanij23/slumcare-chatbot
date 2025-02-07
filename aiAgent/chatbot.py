import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import streamlit as st

class AIAgent:
    def __init__(self,
                 category_model_path='aiagent/category_model.pkl',
                 authority_model_path='aiagent/authority_model.pkl',
                 category_encoder_path='aiagent/category_encoder.pkl',
                 authority_encoder_path='aiagent/authority_encoder.pkl',
                 vectorizer_path='aiagent/vectorizer.pkl'):

        try:
            self.category_model = joblib.load(category_model_path)
            self.authority_model = joblib.load(authority_model_path)
            self.category_encoder = joblib.load(category_encoder_path)
            self.authority_encoder = joblib.load(authority_encoder_path)
            self.vectorizer = joblib.load(vectorizer_path)
            print("Models, encoders, and vectorizer loaded successfully!")
        except FileNotFoundError as e:
            print(f"Error loading models or encoders: {e}. Make sure the files exist in the specified paths.")
            raise
        except Exception as e:
            print(f"An unexpected error occurred during initialization: {e}")
            raise

    def predict_category(self, complaint_text):
        try:
            complaint_vector = self.vectorizer.transform([complaint_text])
            predicted_category_encoded = self.category_model.predict(complaint_vector)[0]
            predicted_category = self.category_encoder.inverse_transform([predicted_category_encoded])[0]
            return predicted_category
        except Exception as e:
            print(f"Error during category prediction: {e}")
            return "Error predicting category"

    def predict_authority(self, complaint_text):
        try:
            complaint_vector = self.vectorizer.transform([complaint_text])
            predicted_authority_encoded = self.authority_model.predict(complaint_vector)[0]
            predicted_authority = self.authority_encoder.inverse_transform([predicted_authority_encoded])[0]
            return predicted_authority
        except Exception as e:
            print(f"Error during authority prediction: {e}")
            return "Error predicting authority"

    def process_complaint(self, complaint_text):
        try:
            category = self.predict_category(complaint_text)
            authority = self.predict_authority(complaint_text)
            return {"category": category, "authority": authority}
        except Exception as e:
            print(f"Error during complaint processing: {e}")
            return {"category": "Error", "authority": "Error"}


# Streamlit Chatbot Interface
def main():
    st.title("SlumCare AI Chatbot")

    # Initialize the AI Agent
    try:
        ai_agent = AIAgent()
        st.session_state.agent_loaded = True  # Set a session state flag
    except Exception as e:
        st.error(f"Failed to initialize AI Agent. Please check the console for details. Error: {e}")
        st.session_state.agent_loaded = False # Set flag to indicate loading failure
        return  # Exit the main function if the agent fails to load

    # Initialize chat history in session state
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Enter your complaint here..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        # Display user message in chat interface
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get the AI Agent's response
        if st.session_state.agent_loaded:  # Check if agent loaded successfully
            try:
                response = ai_agent.process_complaint(prompt)
                category = response['category']
                authority = response['authority']
                bot_response = f"Category: {category}\nAuthority: {authority}"
            except Exception as e:
                bot_response = f"An error occurred while processing your complaint: {e}"
        else:
            bot_response = "AI Agent failed to load.  Please refresh and try again."


        # Add bot message to chat history
        st.session_state.messages.append({"role": "assistant", "content": bot_response})
        # Display bot message in chat interface
        with st.chat_message("assistant"):
            st.markdown(bot_response)


if __name__ == "__main__":
    main()