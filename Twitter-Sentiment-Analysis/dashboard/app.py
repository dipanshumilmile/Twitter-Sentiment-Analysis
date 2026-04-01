import streamlit as st
import pandas as pd
import pickle
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# 🔥 Page Config
st.set_page_config(
    page_title="Twitter Sentiment Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 🔥 Advanced Professional CSS
st.markdown("""
    <style>
    /* Main background */
    .stApp {
        background-color: #0e1117;
    }
    
    /* Custom Card Style */
    .metric-card {
        background-color: #161b22;
        border: 1px solid #30363d;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        transition: transform 0.3s;
    }
    .metric-card:hover {
        transform: translateY(-5px);
        border-color: #58a6ff;
    }
    .metric-label {
        color: #8b949e;
        font-size: 14px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .metric-value {
        color: #ffffff;
        font-size: 32px;
        font-weight: bold;
        margin: 10px 0;
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: #161b22;
        border-right: 1px solid #30363d;
    }
    
    /* Prediction Box styling */
    .stTextArea textarea {
        background-color: #0d1117;
        color: white;
        border: 1px solid #30363d;
    }
    </style>
""", unsafe_allow_html=True)

# ----------------- DATA LOADING -----------------
@st.cache_resource
def load_assets():
    model = pickle.load(open("models/model.pkl", "rb"))
    vectorizer = pickle.load(open("models/vectorizer.pkl", "rb"))
    return model, vectorizer

@st.cache_data
def load_data():
    return pd.read_csv("data/processed/cleaned_tweets.csv")

model, vectorizer = load_assets()
df = load_data()

# ----------------- SIDEBAR -----------------
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/twitter.png", width=80)
    st.title("Sentiment AI")
    st.markdown("---")

    # 🔥 FILTERS
    st.subheader("📊 Filters")

    sentiment_filter = st.multiselect(
        "Select Sentiment",
        options=df["sentiment"].unique(),
        default=df["sentiment"].unique()
    )

    search_text = st.text_input("🔍 Search keyword")

    # 🔷 Apply filters
    filtered_df = df[df["sentiment"].isin(sentiment_filter)]

    if search_text:
        filtered_df = filtered_df[
            filtered_df["clean_text"].str.contains(search_text, case=False, na=False)
        ]

    st.markdown("---")

    # 🔷 Prediction
    st.subheader("🔍 Live Prediction")
    user_input = st.text_area("Analyze a custom tweet:", height=150)

    if st.button("Run Intelligence Check", use_container_width=True):
        if user_input:
            vec = vectorizer.transform([user_input])
            prediction = model.predict(vec)[0]

            # 🔥 Confidence Score
            prob = model.predict_proba(vec).max()

            st.markdown("### Result:")
            st.success(f"{prediction} ({prob*100:.2f}% confidence)")
        else:
            st.info("Waiting for input...")

    st.markdown("---")
    st.caption("🚀 Built with Apache Spark + Scikit-Learn")

# ----------------- MAIN DASHBOARD -----------------
st.markdown("<h1 style='text-align: left; color: white;'>Twitter Sentiment Analytics</h1>", unsafe_allow_html=True)
st.markdown("<p style='color: #8b949e;'>Real-time analysis of social media sentiment trends</p>", unsafe_allow_html=True)

# 🔷 Metrics Row
total = len(df)
pos = (filtered_df["sentiment"]=="Positive").sum()
neg = (filtered_df["sentiment"]=="Negative").sum()
neu = (filtered_df["sentiment"]=="Neutral").sum()

m1, m2, m3, m4 = st.columns(4)

def create_card(column, label, value):
    column.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value:,}</div>
        </div>
    """, unsafe_allow_html=True)

create_card(m1, "Total Processed", total)
create_card(m2, "Positive Feedback", pos)
create_card(m3, "Negative Feedback", neg)
create_card(m4, "Neutral Mentions", neu)

st.markdown("<br>", unsafe_allow_html=True)

# 🔷 Charts Row
c1, c2 = st.columns([6, 4])

sentiment_counts = df["sentiment"].value_counts().reset_index()
sentiment_counts.columns = ['Sentiment', 'Count']

with c1:
    st.markdown("### 📊 Volume Distribution")
    # Custom Plotly Bar Chart
    fig_bar = px.bar(
        sentiment_counts, 
        x='Sentiment', 
        y='Count',
        color='Sentiment',
        color_discrete_map={'Positive': '#238636', 'Negative': '#da3633', 'Neutral': '#8b949e', 'Irrelevant': '#30363d'},
        template="plotly_dark"
    )
    fig_bar.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig_bar, use_container_width=True)

with c2:
    st.markdown("### 🍩 Share of Voice")
    # Custom Plotly Donut Chart
    fig_pie = px.pie(
        sentiment_counts, 
        values='Count', 
        names='Sentiment', 
        hole=0.5,
        color='Sentiment',
        color_discrete_map={'Positive': '#238636', 'Negative': '#da3633', 'Neutral': '#8b949e', 'Irrelevant': '#30363d'},
        template="plotly_dark"
    )
    fig_pie.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig_pie, use_container_width=True)

# 🔷 Data Preview Section
with st.expander("📂 View Raw Processed Data"):
    st.dataframe(df.head(100), use_container_width=True)
    st.markdown("### ☁️ Word Cloud (Top Keywords)")

text_data = " ".join(filtered_df["clean_text"].dropna())

if text_data:
    wc = WordCloud(
        width=800,
        height=400,
        background_color='black',
        colormap='viridis'
    ).generate(text_data)

    fig, ax = plt.subplots()
    ax.imshow(wc, interpolation='bilinear')
    ax.axis("off")

    st.pyplot(fig)
else:
    st.info("No data available for word cloud")