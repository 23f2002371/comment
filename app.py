import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from utils.preprocessing import clean_text
from utils.labels import label_map

# -------------------- PAGE CONFIG --------------------

st.set_page_config(
    page_title="Comment Category Classifier",
    page_icon="💬",
    layout="wide"
)

# -------------------- LOAD MODEL --------------------

@st.cache_resource
def load_model():
    return joblib.load("logistic_model.pkl")

model = load_model()

# -------------------- SIDEBAR --------------------

st.sidebar.title("📌 About")

st.sidebar.write("""
### Comment Category Classifier

Built using:

- Logistic Regression
- TF-IDF
- Scikit-learn
- Streamlit

Dataset:

198,000+ News Comments

Developer:

**Rahul Burdak**
""")

st.sidebar.markdown("---")

st.sidebar.info(
    "This model predicts the category of a news comment using Machine Learning."
)

# -------------------- MAIN --------------------

st.title("💬 Comment Category Classifier")

st.write(
    "Predict the category of any news comment using a trained Machine Learning model."
)

comment = st.text_area(
    "Enter Comment",
    height=180,
    placeholder="Type or paste any comment..."
)

# -------------------- PREDICT --------------------

if st.button("Predict Category", use_container_width=True):

    if comment.strip() == "":
        st.warning("Please enter a comment.")
        st.stop()

    comment = clean_text(comment)

    input_df = pd.DataFrame({
        "comment_clean":[comment],
        "emoticon_1":[0],
        "emoticon_2":[0],
        "emoticon_3":[0],
        "upvote":[0],
        "downvote":[0],
        "if_1":[False],
        "if_2":[False],
        "race":["none"],
        "religion":["none"],
        "gender":["none"],
        "disability":[0]
    })

    prediction = model.predict(input_df)[0]

    probabilities = model.predict_proba(input_df)[0]

    confidence = probabilities.max() * 100

    st.markdown("---")

    st.subheader("Prediction")

    st.success(label_map[prediction])

    st.metric(
        "Confidence",
        f"{confidence:.2f}%"
    )

    st.progress(float(confidence/100))

    st.markdown("---")

    st.subheader("Probability Distribution")

    fig, ax = plt.subplots(figsize=(7,4))

    ax.bar(
        list(label_map.values()),
        probabilities
    )

    ax.set_ylim(0,1)

    st.pyplot(fig)