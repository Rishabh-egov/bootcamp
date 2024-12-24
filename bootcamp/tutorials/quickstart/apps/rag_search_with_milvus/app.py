import os
import streamlit as st

st.set_page_config(layout="wide")

from encoder import emb_text
from milvus_utils import get_milvus_client, get_search_results
from ask_llm import get_llm_answer, OpenAI

from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
from pymilvus import MilvusClient


load_dotenv()
COLLECTION_NAME = os.getenv("COLLECTION_NAME")
MILVUS_ENDPOINT = os.getenv("MILVUS_ENDPOINT")
MILVUS_TOKEN = os.getenv("MILVUS_TOKEN")


# Logo
st.image("./pics/Milvus_Logo_Official.png", width=200)

st.markdown(
    """
    <style>
    .title {
        text-align: center;
        font-size: 50px;
        font-weight: bold;
        margin-bottom: 20px;
    }
    .description {
        text-align: center;
        font-size: 18px;
        color: gray;
        margin-bottom: 40px;
    }
    </style>
    <div class="title">RAG Demo</div>
    <div class="description">
        This chatbot is built with Milvus vector database, supported by mxbai text embedding model.<br>
        It supports conversation based on knowledge from DIGIT.
    </div>
    """,
    unsafe_allow_html=True,
)

# Get clients
# milvus_client = get_milvus_client(uri=MILVUS_ENDPOINT, token=MILVUS_TOKEN)
milvus_client = MilvusClient(MILVUS_ENDPOINT)

os.environ["OPENAI_API_KEY"] = ""
openai_client = OpenAI()
model = SentenceTransformer("mixedbread-ai/mxbai-embed-large-v1")

retrieved_lines_with_distances = []

index_params = milvus_client.prepare_index_params()

index_params.add_index(
    field_name="vector",
    metric_type="COSINE",
    index_type="FLAT",
    index_name="vector_index",
    # params={ "nlist": 128 }
)

milvus_client.create_index(
    collection_name="digit_collection_v10",
    index_params=index_params,
    sync=False
)


with st.form("my_form"):
    question = st.text_area("Enter your question:")

    # Sample question: what is the hardware requirements specification if I want to build Milvus and run from source code?
    submitted = st.form_submit_button("Submit")

    if question and submitted:
        # generate the embeddings
        query_vector = model.encode([question])

        # Search the DB for answers
        res = milvus_client.search(collection_name="digit_collection_v10", data=query_vector, limit=50, output_fields=["text"])
        print(res)
        # Stitch together the context
        context = ""
        for i in range(0, len(res[0]), 1):
            context += f"{res[0][i]['entity']['text']}\n"


        SYSTEM_PROMPT = """Human: You are an AI assistant. You are able to find answers to the questions from the contextual passage snippets provided and the technological knowledge you have."""
        USER_PROMPT = f"""Use the following pieces of information enclosed in <context> tags to provide an answer to the question enclosed in <question> tags.<context>{context}</context><question>{question}</question>"""
        # answer = get_llm_answer(openai_client, context, question)

        response = openai_client.chat.completions.create(model="gpt-4o-mini",messages=[{"role": "system", "content": SYSTEM_PROMPT},{"role": "user", "content": USER_PROMPT},],)
        # response
        print(response.choices[0].message.content)
        
        # Display the question and response in a chatbot-style box
        st.chat_message("user").write(question)
        st.chat_message("assistant").write(response.choices[0].message.content)


# Display the retrieved lines in a more readable format
st.sidebar.subheader("Retrieved Lines with Distances:")
for idx, (line, distance) in enumerate(retrieved_lines_with_distances, 1):
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**Result {idx}:**")
    st.sidebar.markdown(f"> {line}")
    st.sidebar.markdown(f"*Distance: {distance:.2f}*")