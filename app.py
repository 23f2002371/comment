import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

from utils.preprocessing import clean_text
from utils.labels import label_map


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Comment Category Classifier",
    page_icon="💬",
    layout="wide"
)


# ============================================================
# LOAD MODEL ARTIFACT
# ============================================================

@st.cache_resource
def load_model():
    model_path = Path(__file__).resolve().parent / "logistic_model.pkl"

    if not model_path.exists():
        st.error("❌ logistic_model.pkl not found.")
        st.stop()

    return joblib.load(model_path)


artifact = load_model()


# ============================================================
# EXTRACT MODEL COMPONENTS
# ============================================================

tfidf = artifact["tfidf"]
selector = artifact["selector"]
scaler = artifact["scaler"]
encoder = artifact["encoder"]
classifier = artifact["model"]

num_cols = artifact["num_cols"]
cat_cols = artifact["cat_cols"]


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("📌 About")

st.sidebar.write("""
### Comment Category Classifier

Built using:

- Logistic Regression
- TF-IDF
- SelectKBest + Chi-Square
- StandardScaler
- One-Hot Encoding
- Scikit-learn
- Streamlit

Dataset:

**198,000+ News Comments**

Developer:

**Rahul Burdak**
""")

st.sidebar.markdown("---")

st.sidebar.info(
    "This model predicts the category of a news comment using "
    "Machine Learning."
)


# ============================================================
# MAIN TITLE
# ============================================================

st.title("💬 Comment Category Classifier")

st.write(
    "Predict the category of any news comment using a trained "
    "Machine Learning model."
)


# ============================================================
# COMMENT INPUT
# ============================================================

st.subheader("Comment details")

with st.form("comment_form"):

    comment = st.text_area(
        "Comment",
        height=180,
        placeholder="Type or paste any comment..."
    )

    st.caption(
        "The metadata fields are optional. Leave unknown values "
        "at their defaults."
    )

    # --------------------------------------------------------
    # NUMERICAL FEATURES
    # --------------------------------------------------------

    upvote = st.number_input(
        "Upvotes",
        min_value=0,
        value=0,
        step=1
    )

    downvote = st.number_input(
        "Downvotes",
        min_value=0,
        value=0,
        step=1
    )

    disability = st.number_input(
        "Disability",
        min_value=0,
        value=0,
        step=1
    )

    # --------------------------------------------------------
    # CATEGORICAL FEATURES
    # --------------------------------------------------------

    race = st.selectbox(
        "Race",
        [
            "none",
            "asian",
            "black",
            "latino",
            "other",
            "unknown",
            "white"
        ]
    )

    religion = st.selectbox(
        "Religion",
        [
            "none",
            "atheist",
            "buddhist",
            "christian",
            "hindu",
            "jewish",
            "muslim",
            "other",
            "unknown"
        ]
    )

    gender = st.selectbox(
        "Gender",
        [
            "none",
            "female",
            "male",
            "other",
            "transgender",
            "unknown"
        ]
    )

    submitted = st.form_submit_button(
        "Predict Category",
        use_container_width=True
    )


# ============================================================
# PREDICTION
# ============================================================

if submitted:

    # --------------------------------------------------------
    # VALIDATE COMMENT
    # --------------------------------------------------------

    if comment.strip() == "":
        st.warning("Please enter a comment.")
        st.stop()

    try:

        # ====================================================
        # 1. CLEAN TEXT
        # ====================================================

        cleaned_comment = clean_text(comment)


        # ====================================================
        # 2. TEXT FEATURES
        # ====================================================

        text_input = tfidf.transform([cleaned_comment])

        text_input = selector.transform(text_input)


        # ====================================================
        # 3. NUMERICAL FEATURES
        # ====================================================

        # Apply exactly the same log transformation
        # used during training.

        input_num = pd.DataFrame({
            "emoticon_1": [0],
            "emoticon_2": [0],
            "emoticon_3": [0],
            "upvote": [upvote],
            "downvote": [downvote],
            "if_1": [0],
            "if_2": [0],
            "disability": [disability]
        })

        for col in [
            "if_1",
            "if_2",
            "upvote",
            "downvote"
        ]:
            input_num[col] = np.log1p(input_num[col])

        input_num = input_num[num_cols]

        input_num = scaler.transform(input_num)


        # ====================================================
        # 4. CATEGORICAL FEATURES
        # ====================================================

        input_cat = pd.DataFrame({
            "race": [race],
            "religion": [religion],
            "gender": [gender]
        })

        input_cat = encoder.transform(input_cat)


        # ====================================================
        # 5. COMBINE FEATURES
        # ====================================================

        from scipy.sparse import hstack

        transformed_input = hstack([
            text_input,
            input_num,
            input_cat
        ]).tocsr()


        # ====================================================
        # 6. PREDICTION
        # ====================================================

        probabilities = classifier.predict_proba(
            transformed_input
        )[0]

        prediction = classifier.classes_[
            probabilities.argmax()
        ]

        confidence = probabilities.max() * 100


        # ====================================================
        # 7. DISPLAY RESULT
        # ====================================================

        st.markdown("---")

        st.subheader("🎯 Prediction")

        predicted_label = label_map.get(
            int(prediction),
            str(prediction)
        )

        st.success(predicted_label)

        st.metric(
            "Confidence",
            f"{confidence:.2f}%"
        )

        st.progress(
            float(confidence / 100)
        )


        # ====================================================
        # 8. PROBABILITY DISTRIBUTION
        # ====================================================

        st.markdown("---")

        st.subheader("📊 Probability Distribution")

        classes = classifier.classes_

        probability_df = pd.DataFrame({
            "Category": [
                label_map.get(
                    int(category),
                    str(category)
                )
                for category in classes
            ],
            "Probability": probabilities
        }).set_index("Category")

        st.bar_chart(
            probability_df,
            y="Probability"
        )


    except Exception as e:

        st.error("❌ Prediction failed.")

        st.exception(e)
