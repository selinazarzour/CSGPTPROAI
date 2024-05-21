import streamlit as st
import replicate
import os
from transformers import AutoTokenizer
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI
from PIL import Image
from pix2tex.cli import LatexOCR

# Display logo
logo_path = "./logo.png"
st.image(logo_path, use_column_width=True)

# Init image to text model
model = LatexOCR()

replicate_api = "r8_YKpM3UBNeDxUjwRhNrdOcnTApcPundN2qAYJP"

client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key="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"
)

# Dictionary to map course names to CSV file paths
course_csv_files = {
    "UW": {
        "CS 135": "cs135-uw.csv",
        "CS 136": "cs136.csv",
        "MATH 135": "math135-uw.csv",
        "MATH 136": "math136-uw.csv",
        "MATH 137": "math137-uw.csv",
        "MATH 138": "math138-uw.csv",
    },
    "UOFT": {
        "APS105": "aps105-uoft.csv",
        "MAT187": "mat187-uoft.csv",
        "MAT188": "mat188-uoft.csv",
    }
}

# Sidebar for selecting university
st.sidebar.write("### Choose University:")
selected_university = st.sidebar.selectbox("", ["UW", "UOFT"])

# Sidebar for selecting course
st.sidebar.write("### Choose Course:")
if selected_university == "UW":
    selected_course = st.sidebar.selectbox("", list(course_csv_files["UW"].keys()))
elif selected_university == "UOFT":
    selected_course = st.sidebar.selectbox("", list(course_csv_files["UOFT"].keys()))

# Load data based on selected university and course
CSV_FILE_PATH = course_csv_files[selected_university][selected_course]
df = pd.read_csv(CSV_FILE_PATH)

# Create a TF-IDF vectorizer and fit it on the questions
vectorizer = TfidfVectorizer().fit(df['Question'])

def retrieve_relevant_answer(question, df, vectorizer):
    # Transform the question and the DataFrame questions to TF-IDF vectors
    question_vec = vectorizer.transform([question])
    df_question_vecs = vectorizer.transform(df['Question'])

    # Compute cosine similarities between the question and all DataFrame questions
    similarities = cosine_similarity(question_vec, df_question_vecs).flatten()

    # Get the index of the most similar question
    most_similar_idx = similarities.argmax()

    return df.iloc[most_similar_idx]['Answer']

def generate_response(question, df, vectorizer, extracted_text):
    retrieved_answer = retrieve_relevant_answer(question, df, vectorizer)
    prompt = f"We have provided context information below.\n---------------------\n{retrieved_answer}---------------------\nGiven this information, please answer the question: {question} {extracted_text}\ If it is not supported by the context above, be inspired and give an answer.\n"
    # print('THE PROMPTTTTTT YOUPIIIIII', prompt)
    completion = client.chat.completions.create(
        model="snowflake/arctic",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.5,
        top_p=1,
        max_tokens=1024,
        stream=True
    )

    response = ""
    for chunk in completion:
        if chunk.choices[0].delta.content is not None:
            response += chunk.choices[0].delta.content

    return response.strip(), prompt

# Set assistant icon to Snowflake logo
icons = {"assistant": "./logosmall.png", "user": "⛷️"}

# App title
# st.set_page_config(page_title="CS GPT Pro")

# Replicate Credentials
with st.sidebar:
    st.title('CS GPT PRO! :)')
    if 'REPLICATE_API_TOKEN' in st.secrets:
        replicate_api = st.secrets['REPLICATE_API_TOKEN']
    else:
        replicate_api = st.text_input('Enter Replicate API token:', type='password')
        if not (replicate_api.startswith('r8_') and len(replicate_api) == 40):
            st.warning('Please enter your Replicate API token.', icon='⚠️')
            st.markdown("**Don't have an API token?** Head over to [Replicate](https://replicate.com) to sign up for one.")

    os.environ['REPLICATE_API_TOKEN'] = replicate_api
    st.subheader("Adjust model parameters")
    temperature = st.sidebar.slider('temperature', min_value=0.01, max_value=5.0, value=0.3, step=0.01)
    top_p = st.sidebar.slider('top_p', min_value=0.01, max_value=1.0, value=0.9, step=0.01)

# Store LLM-generated responses
if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": "Hi. I'm CS GPT Pro, a new, efficient, intelligent, and truly open language model created by the G12 team using Snowflake AI Research. Ask me anything."}]

# Display or clear chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"], avatar=icons[message["role"]]):
        st.write(message["content"])

def clear_chat_history():
    st.session_state.messages = [{"role": "assistant", "content": "Hi. I'm CS GPT Pro, a new, efficient, intelligent, and truly open language model created by the G12 team using Snowflake AI Research. Ask me anything."}]

st.sidebar.button('Clear chat history', on_click=clear_chat_history)
st.sidebar.caption('Built by team G12! Spreading love everywhere')
st.sidebar.caption('check us out at g12uni.com')

@st.cache_resource(show_spinner=False)
def get_tokenizer():
    """Get a tokenizer to make sure we're not sending too much text to the Model."""
    return AutoTokenizer.from_pretrained("huggyllama/llama-7b")

def get_num_tokens(prompt):
    """Get the number of tokens in a given prompt"""
    tokenizer = get_tokenizer()
    tokens = tokenizer.tokenize(prompt)
    return len(tokens)

test = []
def generate_arctic_response(extracted_text):
    prompt = []
    for dict_message in st.session_state.messages:
        if dict_message["role"] == "user":
            prompt.append("user\n" + dict_message["content"] + "")
        else:
            prompt.append("assistant\n" + dict_message["content"] + "")

    prompt.append("assistant")
    prompt.append("")
    prompt_str = "\n".join(prompt)

    if get_num_tokens(prompt_str) >= 3072:
        st.error("Conversation length too long. Please keep it under 3072 tokens.")
        st.button('Clear chat history', on_click=clear_chat_history, key="clear_chat_history")
        st.stop()

    user_question = st.session_state.messages[-1]["content"]
    response, to_print = generate_response(user_question, df, vectorizer, extracted_text)
    # print('THE PROMPT: ', to_print)
    return response




# User-provided prompt
if prompt := st.chat_input(disabled=not replicate_api):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="⛷️"):
        st.write(prompt)
        

    # Generate response directly from user's input question
    with st.chat_message("assistant", avatar="./logosmall.png"):
        # response = generate_response(prompt, df, vectorizer, "")
        if "extracted_text" in st.session_state:
            response = generate_arctic_response(st.session_state.extracted_text)
        else:
            response = generate_arctic_response(st.session_state)
        st.write(response)
    st.session_state.messages.append({"role": "assistant", "content": response})


# Only generate response if there's extracted text from an uploaded image
# if "extracted_text" in st.session_state:
#     with st.chat_message("assistant", avatar="./logosmall.png"):
#         response = generate_arctic_response(st.session_state.extracted_text)
#         st.write(response)
#     st.session_state.messages.append({"role": "assistant", "content": response})
# else:
#     st.error("Please upload an image or provide a prompt.")

# Custom CSS to style the Streamlit elements including sidebar
custom_css = """
<style>
   [data-testid="stChatMessageContent"] {
    background-color: #050A30;
    color: white;
   }

#    [data-testid="stImage"] {
#     width: 50px;
#     height: 50px;
#    }

   [data-testid="stApp"] {
    background-color: #050A30;
    color: white;
   }

   [data-testid="stHeader"] {
    background-color: #050A30;
    color: white;
   }

   [data-testid="stBottomBlockContainer"] {
    background-color: #050A30;
    color: white;
    min-width: 100%;
   }

   [data-testid="stBottom"]{
    background-color: #050A30;
    color: white;
    
   }

   [data-testid="stWidgetLabel"]{
    background-color: #050A30;
    color: white;
   }
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# Upload function that will let the user upload an image on the image png, jpeg folder in the react app
# it will be saved only in his side (frontend)
# then we should be able to access it

"""
Upload an image to get the answer for all your questions!
"""

# Define the folder where images will be saved
image_folder = 'images'

# Ensure the folder exists
if not os.path.exists(image_folder):
    os.makedirs(image_folder)

# Initialize session state to store uploaded file paths
if 'uploaded_file_paths' not in st.session_state:
    st.session_state.uploaded_file_paths = []

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # To read file as bytes:
    bytes_data = uploaded_file.read()
    
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    
    # Save the image to the images folder with a unique identifier
    unique_filename = f"{st.session_state.get('run_id', 0)}_{uploaded_file.name}"
    save_path = os.path.join(image_folder, unique_filename)
    with open(save_path, "wb") as f:
        f.write(bytes_data)

    # Store the file path in session state
    st.session_state.uploaded_file_paths.append(save_path)

    # st.success(f"Saved file: {save_path}")

    img = Image.open(save_path)

    # Extract text from the image
    text_of_image = model(img)
    st.session_state.extracted_text = text_of_image

    # Display the extracted text
    st.write(f"Text has been successfully extracted... write your question")

    # # Use the extracted text to generate a response
    # response = generate_response(text_of_image, df, vectorizer)

    # # Display the response
    # st.write(f"Response: {response}")

    # st.write(f"Response: {output}")
