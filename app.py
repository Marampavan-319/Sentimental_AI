import streamlit as st
from transformers import pipeline
import pandas as pd
import plotly.express as px

# --- 1. SETUP THE PAGE ---
st.set_page_config(page_title="Brand Sentiment Monitor", layout="wide")
st.title("📊 AI Brand Sentiment Monitor")
st.markdown("Analyze customer feedback instantly using HuggingFace Transformers.")

# --- 2. LOAD THE AI MODEL (Cached so it doesn't reload every time) ---
@st.cache_resource
def load_model():
    # This downloads a specific model that is good at sentiment analysis
    return pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

with st.spinner("Loading AI Model... (This happens only once)"):
    analyzer = load_model()

# --- 3. SIDEBAR (User Options) ---
st.sidebar.header("Select Mode")
mode = st.sidebar.radio("Choose Input Type:", ["Single Text Analysis", "Bulk CSV Analysis"])

# --- 4. MODE A: SINGLE TEXT ANALYSIS ---
if mode == "Single Text Analysis":
    st.subheader("📝 Analyze Single Review")
    user_input = st.text_area("Paste a customer review here:")
    
    if st.button("Analyze Sentiment"):
        if user_input:
            # The AI does the work here
            result = analyzer(user_input)[0] 
            label = result['label']
            score = result['score']
            
            # Display Result
            if label == "POSITIVE":
                st.success(f"**Sentiment: POSITIVE** (Confidence: {score:.2f})")
            else:
                st.error(f"**Sentiment: NEGATIVE** (Confidence: {score:.2f})")
        else:
            st.warning("Please enter some text first.")

# --- 5. MODE B: BULK CSV ANALYSIS ---
elif mode == "Bulk CSV Analysis":
    st.subheader("📂 Upload Customer Data")
    uploaded_file = st.file_uploader("Upload a CSV file (Must have a 'Review' column)", type=["csv"])
    
    if uploaded_file is not None:
        # Read the CSV
        df = pd.read_csv(uploaded_file)
        
        # Check if 'Review' column exists
        if "Review" in df.columns:
            st.write("Analyzing data... please wait.")
            
            # Run AI on every row (This applies the model to your data)
            # We map POSITIVE/NEGATIVE to simpler names for the chart
            df['Sentiment_Result'] = df['Review'].apply(lambda x: analyzer(str(x))[0]['label'])
            
            # Show the data table
            st.dataframe(df)
            
            # --- VISUALIZATION (The "Wow" Factor) ---
            col1, col2 = st.columns(2)
            
            # Chart 1: Pie Chart of Positive vs Negative
            with col1:
                st.subheader("Overall Sentiment")
                fig_pie = px.pie(df, names='Sentiment_Result', title='Sentiment Distribution', 
                                 color='Sentiment_Result',
                                 color_discrete_map={'POSITIVE':'green', 'NEGATIVE':'red'})
                st.plotly_chart(fig_pie)
            
            # Chart 2: Bar Chart (Count)
            with col2:
                st.subheader("Sentiment Count")
                fig_bar = px.bar(df, x='Sentiment_Result', color='Sentiment_Result', 
                                 title='Count of Reviews',
                                 color_discrete_map={'POSITIVE':'green', 'NEGATIVE':'red'})
                st.plotly_chart(fig_bar)
                
        else:
            st.error("Error: The CSV file must have a column named 'Review'.")