import pandas as pd
import plotly.express as px
import streamlit as st

from src.config import APP_DESCRIPTION, APP_TITLE, MAX_REVIEW_CHARACTERS
from src.models.model_loader import load_sentiment_model
from src.services.sentiment_analyzer import analyze_many, analyze_text
from src.utils.text_processor import ReviewValidationError

st.set_page_config(
    page_title=APP_TITLE,
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("📊 Sentiment AI 2.0")
st.caption(APP_DESCRIPTION)
st.info(
    "Baseline model: DistilBERT fine-tuned on SST-2. It predicts positive or "
    "negative sentiment; it does not currently provide neutral or emotion labels."
)

with st.sidebar:
    st.header("Analysis settings")
    mode = st.radio(
        "Choose an analysis mode",
        ["Single review", "Bulk CSV"],
    )
    st.divider()
    st.caption("AI model")
    st.code("distilbert-base-uncased-finetuned-sst-2-english", language="text")
    st.caption("The model is downloaded from Hugging Face on first use and cached locally.")

@st.cache_resource(show_spinner="Loading the sentiment model from Hugging Face...")
def get_analyzer():
    return load_sentiment_model()

try:
    analyzer = get_analyzer()
except Exception as exc:
    st.error("The AI model could not be loaded.")
    st.exception(exc)
    st.stop()

if mode == "Single review":
    st.subheader("Analyze one customer review")
    review = st.text_area(
        "Enter a review",
        placeholder="Example: The product arrived quickly and works perfectly!",
        height=150,
        max_chars=MAX_REVIEW_CHARACTERS,
    )

    if st.button("Analyze sentiment", type="primary"):
        try:
            result = analyze_text(review, analyzer)
            left, right = st.columns(2)
            left.metric("Predicted sentiment", result["label"])
            right.metric("Model confidence", f'{result["score"]:.1%}')

            if result["label"] == "POSITIVE":
                st.success("The model classified this review as positive.")
            else:
                st.error("The model classified this review as negative.")

            st.caption(
                "Confidence is the model's score for its selected label, "
                "not a guarantee that the prediction is correct."
            )
        except ReviewValidationError as exc:
            st.warning(str(exc))
        except Exception as exc:
            st.error("Prediction failed. Check the review and try again.")
            st.exception(exc)

else:
    st.subheader("Analyze a CSV file")
    st.write("Upload a CSV containing a column named **Review**.")
    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

    if uploaded_file is not None:
        try:
            dataframe = pd.read_csv(uploaded_file)
            if "Review" not in dataframe.columns:
                st.error("The CSV must contain a column named 'Review'.")
                st.stop()

            if dataframe.empty:
                st.warning("The uploaded CSV contains no rows.")
                st.stop()

            if len(dataframe) > 1000:
                st.warning("For this first version, please upload at most 1,000 reviews.")
                st.stop()

            with st.spinner(f"Analyzing {len(dataframe)} reviews..."):
                predictions = analyze_many(
                    dataframe["Review"].tolist(),
                    analyzer,
                )

            results = pd.DataFrame(predictions)
            output = dataframe.copy()
            output["Sentiment"] = results["label"]
            output["Confidence"] = results["score"]

            positive_count = int((output["Sentiment"] == "POSITIVE").sum())
            negative_count = int((output["Sentiment"] == "NEGATIVE").sum())
            total_count = len(output)

            metric1, metric2, metric3 = st.columns(3)
            metric1.metric("Total reviews", total_count)
            metric2.metric("Positive reviews", positive_count)
            metric3.metric("Negative reviews", negative_count)

            chart_col1, chart_col2 = st.columns(2)
            sentiment_counts = (
                output["Sentiment"].value_counts().rename_axis("Sentiment").reset_index(name="Reviews")
            )

            with chart_col1:
                st.plotly_chart(
                    px.pie(
                        sentiment_counts,
                        names="Sentiment",
                        values="Reviews",
                        title="Sentiment distribution",
                        color="Sentiment",
                        color_discrete_map={"POSITIVE": "#16a34a", "NEGATIVE": "#dc2626"},
                    ),
                    use_container_width=True,
                )

            with chart_col2:
                st.plotly_chart(
                    px.bar(
                        sentiment_counts,
                        x="Sentiment",
                        y="Reviews",
                        title="Review count by sentiment",
                        color="Sentiment",
                        color_discrete_map={"POSITIVE": "#16a34a", "NEGATIVE": "#dc2626"},
                    ),
                    use_container_width=True,
                )

            st.subheader("Analyzed reviews")
            selected_sentiment = st.selectbox(
                "Filter by sentiment",
                ["ALL", "POSITIVE", "NEGATIVE"],
            )
            filtered = output if selected_sentiment == "ALL" else output[
                output["Sentiment"] == selected_sentiment
            ]
            st.dataframe(filtered, use_container_width=True, hide_index=True)

            csv_bytes = output.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download analyzed CSV",
                data=csv_bytes,
                file_name="sentiment_ai_results.csv",
                mime="text/csv",
            )

        except pd.errors.EmptyDataError:
            st.error("The uploaded file is empty.")
        except UnicodeDecodeError:
            st.error("The CSV encoding could not be read. Save it as UTF-8 and try again.")
        except ReviewValidationError as exc:
            st.warning(str(exc))
        except Exception as exc:
            st.error("The CSV could not be analyzed.")
            st.exception(exc)
