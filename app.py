from string import punctuation
import streamlit as st
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import joblib
import streamlit_extras
from streamlit_extras.let_it_rain import rain 
import time


# Load the model
model = joblib.load('review_detector.joblib')

# Set the page configuration
st.set_page_config(page_title="Hotel Reviewer", page_icon="🏨")

# Set the title and description
st.markdown("<h1 style='text-align: center; font-size: 5vw;'>Hotel Review Sentiment Analysis 🏨</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 20px;'>Unlock Your Review's Emotion: Explore our tool that deciphers hotel reviews, revealing positivity or negativity with just a few words.</p>", unsafe_allow_html=True)
st.markdown("<h2></h2>", unsafe_allow_html=True)
# Split the screen into two columns
col1, col2 = st.columns(2)

# Form for user input in the first column
with col1:
    with st.form(key='prediction_form'):
        # Text input for the review
        name = st.text_area("Enter your review", key='review_input', placeholder="Minimum 20 words")
        # Button to trigger prediction
        submit_button = st.form_submit_button(label='Predict')
        if submit_button:
            if len(name.split()) < 20:
                st.error("Please enter at least 20 words.")
        

# Function to clean text
def clean_text(text):
    # Stopwords such as 'and', 'the' are stored in the stop variable
    stop = set(stopwords.words('english'))
    # Punctuation such as . , are stored in the punc variable
    punc = set(punctuation)
    # Create an instance for the WordNetLemmatizer class from the NLTK library
    lemma = WordNetLemmatizer()
    # Tokenization to split text into individual words or tokens.
    tokens = word_tokenize(text)
    # Filter out non-alphabetic characters
    word_tokens = [t for t in tokens if t.isalpha()]
    # Lowercase and lemmatize words that are not in the union of two sets: stop and punc 
    clean_tokens = [lemma.lemmatize(t.lower()) for t in word_tokens if t not in stop.union(punc)]
    # Join the cleaned tokens into a single string
    cleaned_text = ' '.join(clean_tokens)
    return cleaned_text  # You need to return the cleaned text

# Function to predict and display result
def predict(name):
    cleaned_name = clean_text(name)
    prob = model.predict_proba([cleaned_name])
    positive_probability = prob[0][1]  # Probability of getting positive
    negative_probability = prob[0][0]  # Probability of getting negative
    return positive_probability, negative_probability

with col2:
    st.markdown("<b style='font-size: 25px;'>Prediction Result:</b>", unsafe_allow_html=True)
    # Progress bar for visualization
    progress_bar = st.progress(0)
    my_slot1 = st.empty()


    def status_bar():
        # Predict the sentiment and update progress bar
        positive_probability, negative_probability = predict(name)    
        # After prediction, display the result
        if positive_probability > negative_probability:
            st.markdown(
        """
        <style>
            .stProgress > div > div > div > div {
                background-image: linear-gradient(to right, #2EB62C, #57C84D, #83D475);
            }
        </style>""",
        unsafe_allow_html=True,
        )
            progress_percentage = int(positive_probability * 100)
            for i in range(progress_percentage + 1):
                progress_bar.progress(i)
                time.sleep(0.001)  # Adjust sleep duration for the desired speed of progress bar
            my_slot1.write(f"Positive with the probability of {positive_probability*100:.2f}%")
        else:
            st.markdown(
        """
        <style>
            .stProgress > div > div > div > div {
                background-image: linear-gradient(to right, #C10505, #F1959B, #EA4C46);
            }
        </style>""",
        unsafe_allow_html=True,
        )
            progress_percentage1 = int(negative_probability * 100)
            for i in range(progress_percentage1 + 1):
                progress_bar.progress(i)
                time.sleep(0.001)  # Adjust sleep duration for the desired speed of progress bar
            my_slot1.write(f"Negative with the probability of {negative_probability*100:.2f}%")
    
def Goodfeedback():
        st.success('Thank you for your feedback!')
        st.balloons()

def Badfeedback():
        st.error("We're sorry for the inconvenience.")
        st.snow()


# Check if the form is submitted
if submit_button and len(name.split()) > 19:
    # Trigger prediction when the form is submitted
    status_bar()
    st.button('Satisfied with the answer', on_click=Goodfeedback)
    st.button('Not satisfied with the answer', on_click=Badfeedback)


