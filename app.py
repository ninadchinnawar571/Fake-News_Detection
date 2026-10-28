
import re
import html
import logging
from pathlib import Path
from urllib.parse import quote_plus

import joblib
import requests
import streamlit as st
import pandas as pd

from bs4 import BeautifulSoup
from requests.exceptions import RequestException

# PAGE CONFIGURATION

st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded",
)

MODEL_PATH = Path(__file__).parent / "fake_news_model.joblib"
REQUEST_TIMEOUT = 15

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# CUSTOM STYLING

st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.5rem;
        font-weight: 800;
        color: #2563eb;
        margin-bottom: 0;
    }
    .subtitle {
        color: #64748b;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }
    .result-card {
        padding: 1rem 1.2rem;
        border-radius: 12px;
        margin: 0.6rem 0;
        border: 1px solid #d1d5db;
    }
    .real-card {
        background-color: #ecfdf5;
        color: #065f46;
    }
    .fake-card {
        background-color: #fef2f2;
        color: #991b1b;
    }
    .unknown-card {
        background-color: #fffbeb;
        color: #92400e;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# TEXT PREPROCESSING
# Keep consistent with notebook preprocessing

def clean_text(text):
    """Clean article text without removing useful word information."""
    if pd.isna(text):
        return ""

    text = html.unescape(str(text))
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = BeautifulSoup(text, "html.parser").get_text(" ")

    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


# LOAD SAVED PIPELINE

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH.name}. "
            "Run the notebook training and export cell first."
        )

    return joblib.load(MODEL_PATH)

# PREDICTION
# Label mapping: 1 = Real, 0 = Fake

def predict_news(news, pipeline):
    cleaned = clean_text(news)

    if len(cleaned.split()) < 5:
        return {
            "label": "Insufficient text",
            "score": None,
        }

    prediction = int(pipeline.predict([cleaned])[0])

    score = None

    if hasattr(pipeline, "decision_function"):
        raw_score = pipeline.decision_function([cleaned])
        score = float(raw_score[0])

    label = "Real-like" if prediction == 1 else "Fake-like"

    return {
        "label": label,
        "score": score,
    }



# GOOGLE NEWS RSS

def fetch_live_news(limit=10, query=None):
    if query:
        url = (
            "https://news.google.com/rss/search?q="
            f"{quote_plus(query)}"
            "&hl=en-IN&gl=IN&ceid=IN:en"
        )
    else:
        url = (
            "https://news.google.com/rss"
            "?hl=en-IN&gl=IN&ceid=IN:en"
        )

    try:
        response = requests.get(
            url,
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=REQUEST_TIMEOUT,
        )
        response.raise_for_status()

        soup = BeautifulSoup(response.content, "xml")
        articles = []

        for item in soup.find_all("item")[:limit]:
            title_tag = item.find("title")
            link_tag = item.find("link")
            date_tag = item.find("pubDate")
            source_tag = item.find("source")

            title = (
                title_tag.get_text(" ", strip=True)
                if title_tag else ""
            )
            link = (
                link_tag.get_text(strip=True)
                if link_tag else ""
            )

            if not title:
                continue

            articles.append({
                "Headline": title,
                "Source": (
                    source_tag.get_text(strip=True)
                    if source_tag else "Unknown"
                ),
                "Published": (
                    date_tag.get_text(strip=True)
                    if date_tag else "Unknown"
                ),
                "Model prediction": "",
                "Article link": link,
            })

        return articles

    except RequestException as exc:
        logger.warning("RSS request failed: %s", exc)
        st.error(
            "Could not retrieve live headlines. "
            "Check your internet connection and try again."
        )
        return []

    except Exception:
        logger.exception("RSS parsing failed")
        st.error("Unable to parse the news feed.")
        return []

# SIDEBAR

st.sidebar.title("News Detector")
st.sidebar.caption("Machine-learning news text analysis")

page = st.sidebar.radio(
    "Navigate",
    [
        "Home",
        "Check News",
        "Live News",
        "About Model",
    ],
)

st.sidebar.divider()
st.sidebar.info(
    "Predictions are indicative, not verified fact-checks."
)

# LOAD MODEL

try:
    pipeline = load_model()
    model_available = True
except Exception as exc:
    pipeline = None
    model_available = False

    st.sidebar.error("Trained model unavailable.")
    if page != "About Model":
        st.error(str(exc))
        st.info(
            "Train your model in the notebook and save it as "
            "'fake_news_model.joblib' beside app.py."
        )

# HOME

if page == "Home":
    st.markdown(
        '<div class="main-title">Fake News Detector</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="subtitle">'
        "Analyze news text and explore live headlines using "
        "your trained machine-learning model."
        "</div>",
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Analysis modes", "2")
    with col2:
        st.metric("Live feed", "Google News RSS")
    with col3:
        st.metric(
            "Model status",
            "Ready" if model_available else "Not loaded",
        )

    st.divider()

    st.subheader("What would you like to do?")

    left, right = st.columns(2)

    with left:
        st.markdown("### Check an article")
        st.write(
            "Paste a headline or article to receive a text-model "
            "classification."
        )
        if st.button("Open News Checker", use_container_width=True):
            st.session_state["selected_page"] = "Check News"
            st.rerun()

    with right:
        st.markdown("### Explore live headlines")
        st.write(
            "Retrieve recent headlines and examine the model's "
            "predictions."
        )
        if st.button("Open Live News", use_container_width=True):
            st.session_state["selected_page"] = "Live News"
            st.rerun()

    st.warning(
        "A text classifier cannot independently confirm facts. "
        "Always check the original report and corroborating sources."
    )

# CHECK NEWS

elif page == "Check News":
    st.title("Check a News Article")
    st.write(
        "Paste a news headline or full article for analysis."
    )

     # Create the form and define submitted
    with st.form("news_form"):
        news_text = st.text_area(
            "News headline or article",
            height=250,
            placeholder="Paste your news text here...",
        )

        submitted = st.form_submit_button(
            "Analyze News",
            type="primary",
            use_container_width=True,
        )

    
# Process the form only after clicking the button
    if submitted:
        if not model_available:
            st.error("Train and export the model first.")

        elif not news_text.strip():
            st.warning("Please enter some news text.")

        else:
            with st.spinner("Analyzing text..."):
                result = predict_news(news_text, pipeline)

            st.subheader("Analysis Result")
            st.write("Prediction:", result["label"])

            if result.get("score") is not None:
                st.write(
                    f"SVM decision score: {result['score']:.3f}"
                )

            if result["label"] == "Real-like":
                st.success("Model prediction: Real-like")
            elif result["label"] == "Fake-like":
                st.error("Model prediction: Fake-like")
            else:
                st.warning("Please enter a longer article.")

            st.warning(
                "This is a text-based prediction, not independent "
                "verification of the article's claims."
            )

# LIVE NEWS

elif page == "Live News":
    st.title("Live News Analysis")

    st.write(
        "Retrieve headlines from Google News RSS and classify "
        "their text. Results are not independent fact-checks."
    )

    with st.form("live_news_form"):
        query = st.text_input(
            "Search topic (optional)",
            placeholder="e.g. science, health, elections",
        )
        limit = st.slider(
            "Number of headlines",
            min_value=5,
            max_value=25,
            value=10,
        )
        fetch_clicked = st.form_submit_button(
            "Fetch and Analyze",
            type="primary",
            use_container_width=True,
        )

    if fetch_clicked:
        if not model_available:
            st.error("Train and export the model first.")
        else:
            with st.spinner("Fetching live headlines..."):
                articles = fetch_live_news(
                    limit=limit,
                    query=query.strip() or None,
                )

            if articles:
                for article in articles:
                    result = predict_news(
                        article["Headline"],
                        pipeline,
                    )
                    article["Model prediction"] = result["label"]

                st.session_state["live_articles"] = articles

    articles = st.session_state.get("live_articles", [])

    if articles:
        st.subheader(f"Retrieved headlines: {len(articles)}")

        for index, article in enumerate(articles, start=1):
            with st.expander(
                f"{index}. {article['Headline']}",
                expanded=(index == 1),
            ):
                st.write(f"**Source:** {article['Source']}")
                st.write(f"**Published:** {article['Published']}")

                prediction = article["Model prediction"]

                if prediction == "Real-like":
                    st.success(f"Model prediction: {prediction}")
                elif prediction == "Fake-like":
                    st.error(f"Model prediction: {prediction}")
                else:
                    st.warning(f"Model prediction: {prediction}")

                st.caption(
                    "This result is based on the headline text "
                    "and should not be treated as a fact-check."
                )

                if article["Article link"]:
                    st.link_button(
                        "Open Original News Link",
                        article["Article link"],
                    )

        df = pd.DataFrame(articles)

        st.download_button(
            "Download Results as CSV",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name="news_analysis.csv",
            mime="text/csv",
        )

# ABOUT MODEL

elif page == "About Model":
    st.title("About the Model")

    st.markdown(
        """
        This application is designed to use a trained scikit-learn
        text-classification pipeline.

        **Expected model labels**
        - `1`: Real-like
        - `0`: Fake-like

        **Typical pipeline**
        1. Text cleaning
        2. TF-IDF feature extraction
        3. Classification using a trained algorithm

        **Evaluation recommended**
        - Held-out test accuracy
        - Precision and recall for each class
        - Macro F1-score
        - Confusion matrix
        - Evaluation on newer, unseen news articles

        **Limitations**

        A classifier learns patterns from its training data. It may
        misclassify satire, breaking news, quotations, unfamiliar
        topics, or articles from sources not represented in training.
        """
    )

    if model_available:
        st.success("Saved model pipeline loaded successfully.")
    else:
        st.warning(
            "The saved model is missing or could not be loaded."
        )

    st.caption(
        "Only load trusted joblib/pickle model files, since "
        "loading an untrusted file can execute malicious code."
    )
