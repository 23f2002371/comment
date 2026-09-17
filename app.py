import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from pathlib import Path

from utils.preprocessing import clean_text
from utils.labels import label_map

NEUTRAL_WEIGHT = 0.01

st.set_page_config(
    page_title="Comment Category Classifier",
    page_icon="💬",
    layout="wide"
)

@st.cache_resource
def load_model():
    model_path = Path(__file__).resolve().parent / "logistic_model.pkl"
    with open(model_path, "rb") as model_file:
        return pickle.load(model_file)

model = load_model()

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

st.title("💬 Comment Category Classifier")

st.write(
    "Predict the category of any news comment using a trained Machine Learning model."
)

st.subheader("Comment details")

with st.form("comment_form"):
    comment = st.text_area(
        "Comment",
        height=180,
        placeholder="Type or paste any comment..."
    )

    st.caption("The metadata fields are optional. Leave unknown values at their defaults.")

    upvote = st.number_input("Upvotes", min_value=0, value=0, step=1)
    downvote = st.number_input("Downvotes", min_value=0, value=0, step=1)
    disability = st.number_input("Disability", min_value=0, value=0, step=1)

    race = st.selectbox(
        "Race",
        ["none", "asian", "black", "latino", "other", "unknown", "white"]
    )
    religion = st.selectbox(
        "Religion",
        ["none", "atheist", "buddhist", "christian", "hindu", "jewish", "muslim", "other", "unknown"]
    )
    gender = st.selectbox(
        "Gender",
        ["none", "female", "male", "other", "transgender", "unknown"]
    )

    submitted = st.form_submit_button("Predict Category", use_container_width=True)

if submitted:

    if comment.strip() == "":
        st.warning("Please enter a comment.")
        st.stop()

    input_df = pd.DataFrame({
        "comment_clean": [clean_text(comment)],
        "emoticon_1": [0],
        "emoticon_2": [0],
        "emoticon_3": [0],
        "upvote": [upvote],
        "downvote": [downvote],
        "if_1": [False],
        "if_2": [False],
        "race": [race],
        "religion": [religion],
        "gender": [gender],
        "disability": [disability]
    })

    transformed_input = model.named_steps["prep"].transform(input_df)
    feature_names = model.named_steps["prep"].get_feature_names_out()
    religion_columns = np.char.startswith(feature_names.astype(str), "cat__religion_")
    religion_weights = np.ones(transformed_input.shape[1])
    religion_weights[religion_columns] = 1.5
    transformed_input = transformed_input.multiply(religion_weights)

    classifier = model.named_steps["model"]
    probabilities = classifier.predict_proba(transformed_input)[0]
    adjusted_probabilities = probabilities.copy()
    neutral_index = np.flatnonzero(classifier.classes_ == 0)
    if neutral_index.size:
        adjusted_probabilities[neutral_index[0]] *= NEUTRAL_WEIGHT
        adjusted_probabilities /= adjusted_probabilities.sum()

    prediction = classifier.classes_[adjusted_probabilities.argmax()]

    confidence = adjusted_probabilities.max() * 100

    st.markdown("---")

    st.subheader("Prediction")

    st.success(label_map.get(int(prediction), str(prediction)))

    st.metric(
        "Confidence",
        f"{confidence:.2f}%"
    )

    st.progress(float(confidence/100))

    st.markdown("---")

    st.subheader("Probability Distribution")

    classes = getattr(model, "classes_", range(len(probabilities)))
    probability_df = pd.DataFrame({
        "Category": [label_map.get(int(category), str(category)) for category in classes],
        "Probability": adjusted_probabilities
    })

    figure, axis = plt.subplots(figsize=(9, 4.5))
    bars = axis.barh(
        probability_df["Category"],
        probability_df["Probability"],
        color=["#4CAF50", "#FFC107", "#F44336", "#212121"]
    )
    axis.set_title("Comment Category Confidence")
    axis.set_xlabel("Confidence")
    axis.set_xlim(0, 1)
    axis.xaxis.set_major_formatter(plt.FuncFormatter(lambda value, _: f"{value:.0%}"))
    axis.grid(axis="x", linestyle="--", alpha=0.35)
    axis.set_axisbelow(True)

    for bar, probability in zip(bars, probability_df["Probability"]):
        axis.text(
            min(probability + 0.02, 0.98),
            bar.get_y() + bar.get_height() / 2,
            f"{probability:.2%}",
            va="center"
        )

    figure.tight_layout()
    st.pyplot(figure, use_container_width=True)
    plt.close(figure)
