import streamlit as st
import pandas as pd
import google.generativeai as genai
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet 
import io
import numpy as np
from sklearn.linear_model import LinearRegression
genai.configure(api_key="")
st.title("AI-Powered Retail Sales Analytics Platform")

st.markdown(
    "Analyze sales performance, customer behavior, product trends, and generate AI-powered business recommendations."
)
uploaded_file=st.file_uploader("Upload a CSV File", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
else:
    st.warning("Please upload a CSV file")
    st.stop()
selected_region=st.selectbox("Select Region", ["All"]+list(df["Region"].unique()))
if selected_region!="All":
    df = df[df["Region"]==selected_region]
selected_category=st.selectbox("Select Category",["All"]+list(df["Category"].unique()))
if selected_category!="All":
    df = df[df["Category"]==selected_category]
# df = pd.read_csv("../dataset/superstore_final_dataset.csv")
# st.write(df.columns)
st.subheader("Dataset Preview")
st.dataframe(df.head())
total_sales = df["Sales"].sum()
total_orders = df["Order_ID"].nunique()

top_category=(df.groupby("Category")["Sales"].sum().idxmax())

st.subheader("Business Metrics")
col1, col2, col3= st.columns(3)

with col1:
    st.metric("Total Sales",f"${total_sales:,.0f}")
with col2:
    st.metric("Orders", total_orders)
with col3:
    st.metric("Top Category", top_category)

# total_profit = df["Profit"].sum()

region_sales=(df.groupby("Region")["Sales"].sum().sort_values(ascending=False))

st.subheader("Region-wise Sales")
st.dataframe(region_sales)
st.bar_chart(region_sales)


category_sales=(df.groupby("Category")["Sales"].sum().sort_values(ascending=False))

st.subheader("Category-wise Sales")
st.dataframe(category_sales)
st.bar_chart(category_sales)

top_customers=(df.groupby("Customer_Name")["Sales"].sum().sort_values(ascending=False))

st.subheader("Top 5 Customers")
st.dataframe(top_customers)
st.bar_chart(top_customers)

top_products=(df.groupby("Product_Name")["Sales"].sum().sort_values(ascending=False).head(5))

st.subheader("Top 5 Products")
st.dataframe(top_products)
st.bar_chart(top_products)

df["Order_Date"]=pd.to_datetime(df["Order_Date"], dayfirst=True)
monthly_sales=(df.groupby(df["Order_Date"].dt.to_period("M"))["Sales"].sum())
monthly_sales.index=(monthly_sales.index.astype(str))
st.subheader("Monthly Sales Trend")
st.line_chart(monthly_sales)

st.subheader("Top 10 Customers")
top_customers=(df.groupby("Customer_Name")["Sales"].sum().sort_values(ascending=False).head(10))
st.dataframe(top_customers)
st.bar_chart(top_customers)

st.subheader("Top 10 Products")

top_products = (
    df.groupby("Product_Name")["Sales"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
)

st.dataframe(top_products)
st.bar_chart(top_products)

st.subheader("Sales Forecast")

monthly_sales_df = monthly_sales.reset_index()

monthly_sales_df.columns = ["Month", "Sales"]

monthly_sales_df["Month_Number"] = np.arange(
    len(monthly_sales_df)
)

X = monthly_sales_df[["Month_Number"]]
y = monthly_sales_df["Sales"]

model_lr = LinearRegression()
model_lr.fit(X, y)

next_month = [[len(monthly_sales_df)]]

predicted_sales = model_lr.predict(next_month)[0]

st.metric(
    "Predicted Next Month Sales",
    f"${predicted_sales:,.0f}"
)
st.subheader("Business Health Score")

health_score = 0

# Sales Score
if total_sales > 2000000:
    health_score += 40
elif total_sales > 1000000:
    health_score += 25
else:
    health_score += 10

# Orders Score
if total_orders > 4000:
    health_score += 30
elif total_orders > 2000:
    health_score += 20
else:
    health_score += 10

# Category Score
if top_category == "Technology":
    health_score += 30
else:
    health_score += 20

st.metric(
    "Business Health Score",
    f"{health_score}/100"
)
st.subheader("Ask Questions About Your Data")
user_question = st.text_input("Ask anything about your sales data")
if user_question:
    data_summary = f"""
    Total Sales:{total_sales}
    Total Orders:{total_orders}
    Top Category:{top_category}

    Region Sales:
    {region_sales.to_string()}"""

    question_prompt = f"""
    You are a Business Analyst.

    Dataset Summary:
    {data_summary}

    User Question:
    {user_question}

    Answer the question based on the dataset summary."""

    model=genai.GenerativeModel("gemini-2.5-flash")
    response=model.generate_content(question_prompt)
    st.subheader("AI Answer")
    st.write(response.text)

if st.button("Generate AI Insights"):
    prompt = f"""
You are a Senior Retail Business Analyst.

Analyze this retail sales dataset and provide:

1. Key Business Insights
2. Opportunities
3. Risks
4. Recommendations
5. Growth Strategies

DATA

Total Sales:
{total_sales}

Total Orders:
{total_orders}

Top Category:
{top_category}

Region Sales:
{region_sales.to_string()}

Category Sales:
{category_sales.to_string()}

Top Customers:
{top_customers.to_string()}

Top Products:
{top_products.to_string()}

Monthly Sales:
{monthly_sales.to_string()}
"""

    model= genai.GenerativeModel("gemini-2.5-flash")
    response = model.generate_content(prompt)

    st.write(response.text)
    pdf_buffer = io.BytesIO()
    doc = SimpleDocTemplate(pdf_buffer)
    styles = getSampleStyleSheet()
    clean_text = response.text.replace("**","")
    content=[]

    for line in clean_text.split("\n"):
        if line.strip():
            content.append(Paragraph(line,styles["Normal"]))
    doc.build(content)
    pdf_data = pdf_buffer.getvalue()
    st.download_button(
    label="📄 Download Executive PDF Report",
    data=pdf_data,
    file_name="AI_Retail_Insights.pdf",
    mime="application/pdf"
    )
if st.button("Generate Business Recommendations"):

    recommendation_prompt = f"""
    You are a Senior Business Consultant.

    Based on the following sales analytics:

    Total Sales: {total_sales}

    Total Orders: {total_orders}

    Top Category: {top_category}

    Region Sales:
    {region_sales.to_string()}

    Category Sales:
    {category_sales.to_string()}

    Top Customers:
    {top_customers.to_string()}

    Top Products:
    {top_products.to_string()}

    Provide:

    1. Revenue Growth Strategies
    2. Customer Retention Recommendations
    3. Product Optimization Suggestions
    4. Regional Expansion Opportunities
    5. Key Business Risks
    6. Action Plan for Next Quarter
    """

    model = genai.GenerativeModel("gemini-2.5-flash")

    recommendation_response = model.generate_content(
        recommendation_prompt
    )

    st.subheader("AI Business Recommendations")

    st.write(recommendation_response.text)