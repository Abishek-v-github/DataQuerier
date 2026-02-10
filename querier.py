import streamlit as st
import pandas as pd
import google.generativeai as genai
import os
from io import StringIO

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or st.secrets.get("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    st.error("GEMINI_API_KEY not configured. Add it to Streamlit secrets.")
    st.stop()

genai.configure(api_key=GEMINI_API_KEY)

SAMPLE_CSVS = {
    "Employees Dataset": """
emp_id,name,department,salary,experience
1,Alice,Engineering,80000,5
2,Bob,HR,50000,3
3,Charlie,Engineering,90000,6
4,David,Marketing,60000,4
5,Eva,HR,55000,2
""",
    "Sales Dataset": """
order_id,region,product,revenue,units_sold
101,North,Laptop,120000,40
102,South,Mobile,80000,60
103,East,Laptop,95000,30
104,West,Tablet,50000,25
105,North,Mobile,70000,50
""",
    "Students Dataset": """
student_id,name,course,marks,attendance
1,Rahul,AI,88,92
2,Anita,DS,91,95
3,John,AI,76,85
4,Priya,ML,89,90
5,Karan,DS,82,88
"""
}


def load_csv_from_string(csv_string):
    return pd.read_csv(StringIO(csv_string.strip()))

def ask_gemini(query, context):
    model = genai.GenerativeModel("gemini-2.5-flash")
    prompt = f"""
You are given the following dataset:

{context}

Answer the user's question accurately:
{query}
"""
    response = model.generate_content(prompt)
    return response.text if response else "No response from model."


def main():
    st.set_page_config(page_title="The Querier – CSV Mode")
    st.title("📊 The Querier – CSV Query Playground")

    st.markdown(
        "Use built-in datasets or upload your own CSV and query it using natural language."
    )

    data_source = st.radio(
        "Choose data source:",
        ("Use sample dataset", "Upload my own CSV")
    )

    df = None

    if data_source == "Use sample dataset":
        dataset_name = st.selectbox(
            "Choose a sample CSV dataset:",
            list(SAMPLE_CSVS.keys())
        )
        df = load_csv_from_string(SAMPLE_CSVS[dataset_name])

    else:
        uploaded_file = st.file_uploader(
            "Upload a CSV file",
            type=["csv"]
        )
        if uploaded_file:
            df = pd.read_csv(uploaded_file)

    if df is not None:
        st.subheader("🔍 Data Preview")
        st.dataframe(df.head(20))

        query = st.text_input("Ask a question about this dataset")

        if query:
            with st.spinner("Thinking..."):
                response = ask_gemini(query, df.to_string(index=False))
                st.subheader("🧠 Answer")
                st.write(response)
    else:
        st.info("Please select a dataset or upload a CSV to continue.")

if __name__ == "__main__":
    main()
