import os
import tempfile
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import pyttsx3

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix


st.set_page_config(
    page_title="AI Diabetes Predictor",
    page_icon="🩺",
    layout="wide"
)


st.markdown("""
<style>
.stApp {
    background: radial-gradient(circle at top left, #1e3a8a, #020617 55%);
    color: #e5e7eb;
}
.block-container {
    padding-top: 2rem;
}
.hero {
    background: linear-gradient(135deg, rgba(56,189,248,0.18), rgba(15,23,42,0.95));
    padding: 30px;
    border-radius: 28px;
    border: 1px solid rgba(255,255,255,0.14);
    box-shadow: 0 20px 50px rgba(0,0,0,0.35);
    text-align: center;
    margin-bottom: 25px;
}
.hero-title {
    font-size: 52px;
    font-weight: 900;
    color: #38bdf8;
}
.hero-subtitle {
    font-size: 18px;
    color: #cbd5e1;
}
.developer {
    color: #a5f3fc;
    font-size: 16px;
    margin-top: 10px;
}
.premium-card {
    background: rgba(15,23,42,0.78);
    padding: 24px;
    border-radius: 24px;
    border: 1px solid rgba(255,255,255,0.13);
    box-shadow: 0 12px 35px rgba(0,0,0,0.30);
    margin-bottom: 18px;
}
.chat-user {
    background: linear-gradient(135deg, #1e293b, #334155);
    padding: 12px;
    border-radius: 14px;
    margin-bottom: 10px;
    border-left: 5px solid #38bdf8;
}
.chat-bot {
    background: linear-gradient(135deg, #052e16, #0f172a);
    padding: 12px;
    border-radius: 14px;
    margin-bottom: 10px;
    border-left: 5px solid #22c55e;
}
.stButton > button {
    border-radius: 14px;
    background: linear-gradient(135deg, #0284c7, #2563eb);
    color: white;
    border: none;
    font-weight: 700;
    padding: 0.6rem 1rem;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #0ea5e9, #3b82f6);
    color: white;
}
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #020617, #0f172a);
    border-right: 1px solid rgba(255,255,255,0.10);
}
[data-testid="stMetric"] {
    background: rgba(15,23,42,0.85);
    padding: 20px;
    border-radius: 20px;
    border: 1px solid rgba(255,255,255,0.12);
    box-shadow: 0 10px 30px rgba(0,0,0,0.25);
}
.disclaimer {
    background: rgba(251,191,36,0.12);
    border-left: 6px solid #fbbf24;
    padding: 18px;
    border-radius: 16px;
    color: #fde68a;
    margin-top: 20px;
}
</style>
""", unsafe_allow_html=True)


def speak_text(text):
    try:
        engine = pyttsx3.init()
        engine.setProperty("rate", 160)

        audio_file = tempfile.NamedTemporaryFile(delete=False, suffix=".aiff")
        audio_path = audio_file.name
        audio_file.close()

        engine.save_to_file(text, audio_path)
        engine.runAndWait()

        return audio_path
    except Exception:
        return None


def offline_bot(question, accuracy):
    q = question.lower()

    if "diabetes" in q:
        return "Diabetes is a condition where blood sugar level becomes higher than normal."
    elif "glucose" in q:
        return "Glucose means blood sugar. It is one of the most important features in this project."
    elif "bmi" in q:
        return "BMI means Body Mass Index. It is calculated using height and weight."
    elif "model" in q or "random forest" in q:
        return "This project uses Random Forest Classifier as the main prediction model."
    elif "accuracy" in q:
        return f"The current Random Forest model accuracy is {accuracy * 100:.2f}%."
    elif "risk" in q or "prediction" in q:
        if "last_prediction" in st.session_state and st.session_state.last_prediction:
            p = st.session_state.last_prediction
            return (
                f"Your last prediction was {p['result']} with "
                f"{p['probability']:.2f}% probability. Risk level was {p['risk_level']}."
            )
        return "Risk is calculated using prediction probability from the trained ML model."
    elif "doctor" in q:
        return "If the risk level is medium or high, the system suggests consulting a doctor for proper medical diagnosis."
    elif "precaution" in q or "advice" in q:
        return "Precautions include balanced diet, regular exercise, reducing sugar intake, drinking water, and regular health checkups."
    elif "history" in q:
        return "Patient history stores previous predictions in a CSV file for record keeping."
    elif "confusion" in q:
        return "Confusion matrix shows correct and incorrect predictions made by the model."
    elif "compare" in q:
        return "Model comparison checks multiple ML algorithms and compares their accuracy."
    else:
        return "You can ask about diabetes, glucose, BMI, model, accuracy, risk, doctor advice, precautions, confusion matrix, model comparison, or patient history."


st.markdown("""
<div class="hero">
    <div class="hero-title">🩺 AI Diabetes Risk Predictor</div>
    <div class="hero-subtitle">Premium Machine Learning Dashboard with AI Assistant, Model Analysis & Patient History</div>
    <div class="developer">👨‍💻 Developed by Lakshya Kanodia</div>
</div>
""", unsafe_allow_html=True)


df = pd.read_csv("diabetes.csv")

X = df.drop("Outcome", axis=1)
y = df["Outcome"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


models = {
    "Random Forest": RandomForestClassifier(random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5)
}

model_scores = {}

for name, clf in models.items():
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    model_scores[name] = accuracy_score(y_test, pred)

model = models["Random Forest"]
y_pred = model.predict(X_test)
accuracy = model_scores["Random Forest"]


col1, col2, col3, col4 = st.columns(4)
col1.metric("🎯 Accuracy", f"{accuracy * 100:.2f}%")
col2.metric("📊 Records", len(df))
col3.metric("🧠 Main Model", "Random Forest")
col4.metric("🧪 Models Tested", len(models))


with st.sidebar:
    st.title("🤖 AI Assistant")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if "last_prediction" not in st.session_state:
        st.session_state.last_prediction = None

    if "selected_question" not in st.session_state:
        st.session_state.selected_question = ""

    st.markdown("### 💡 Suggested Questions")

    q1, q2 = st.columns(2)

    with q1:
        if st.button("Diabetes"):
            st.session_state.selected_question = "What is diabetes?"
        if st.button("Model"):
            st.session_state.selected_question = "Which model is used?"
        if st.button("Doctor"):
            st.session_state.selected_question = "When should I consult a doctor?"

    with q2:
        if st.button("Accuracy"):
            st.session_state.selected_question = "What is accuracy?"
        if st.button("Risk"):
            st.session_state.selected_question = "Explain my prediction risk"
        if st.button("Advice"):
            st.session_state.selected_question = "Give precautions"

    question = st.text_input(
        "Ask something...",
        value=st.session_state.get("selected_question", "")
    )

    if st.button("Ask Assistant"):
        if question.strip():
            reply = offline_bot(question, accuracy)

            st.session_state.chat_history.append({"role": "user", "content": question})
            st.session_state.chat_history.append({"role": "assistant", "content": reply})

            audio_path = speak_text(reply)
            if audio_path and os.path.exists(audio_path):
                st.audio(audio_path)

            st.session_state.selected_question = ""

    if st.button("Clear Chat"):
        st.session_state.chat_history = []

    st.markdown("---")
    st.markdown("### 💬 Chat Memory")

    for msg in st.session_state.chat_history[-8:]:
        if msg["role"] == "user":
            st.markdown(
                f"<div class='chat-user'>🧑 <b>You:</b> {msg['content']}</div>",
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"<div class='chat-bot'>🤖 <b>Assistant:</b> {msg['content']}</div>",
                unsafe_allow_html=True
            )


tab1, tab2, tab3, tab4 = st.tabs(
    ["🔍 Prediction", "📊 Visual Analytics", "🧪 Model Analysis", "🗂️ Patient History"]
)


with tab1:
    st.markdown("<div class='premium-card'>", unsafe_allow_html=True)
    st.subheader("📝 Enter Patient Details")

    left, right = st.columns([1, 1])

    with left:
        patient_name = st.text_input("Patient Name", "Unknown")
        preg = st.number_input("Pregnancies", 0, 20, 1)
        glucose = st.number_input("Glucose Level", 0, 250, 120)
        bp = st.number_input("Blood Pressure", 0, 150, 70)
        skin = st.number_input("Skin Thickness", 0, 100, 20)

    with right:
        insulin = st.number_input("Insulin", 0, 900, 80)
        bmi = st.number_input("BMI", 0.0, 70.0, 25.0)
        dpf = st.number_input("Diabetes Pedigree Function", 0.0, 3.0, 0.5)
        age = st.number_input("Age", 1, 120, 25)

    predict_btn = st.button("🔍 Predict Diabetes Risk", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    input_data = np.array([[preg, glucose, bp, skin, insulin, bmi, dpf, age]])

    if predict_btn:
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0]
        diabetes_probability = probability[1] * 100

        st.markdown("<div class='premium-card'>", unsafe_allow_html=True)
        st.subheader("✅ Prediction Result")

        if prediction == 1:
            result_text = "High Risk of Diabetes"
            st.error("🚨 High Risk of Diabetes")
        else:
            result_text = "Low Risk of Diabetes"
            st.success("✅ Low Risk of Diabetes")

        st.write(f"### Diabetes Risk Probability: **{diabetes_probability:.2f}%**")
        st.progress(int(diabetes_probability))

        if diabetes_probability < 30:
            risk_level = "Low"
            st.success("Risk Level: Low")
        elif diabetes_probability < 70:
            risk_level = "Medium"
            st.warning("Risk Level: Medium")
        else:
            risk_level = "High"
            st.error("Risk Level: High")

        st.subheader("🩺 Doctor Advice & Precautions")

        if risk_level == "High":
            doctor_advice = "Please consult a doctor as soon as possible for proper medical diagnosis."
            precautions = """
            - Avoid sugary foods and soft drinks
            - Check blood sugar regularly
            - Do at least 30 minutes of light exercise daily
            - Maintain a balanced diet
            - Drink enough water
            - Avoid junk food and processed food
            - Take proper sleep
            """
            st.error(f"⚠️ Doctor Recommendation: {doctor_advice}")
            st.markdown("### 🛡️ Precautions")
            st.markdown(precautions)

        elif risk_level == "Medium":
            doctor_advice = "You should consider a health checkup and consult a doctor if symptoms appear."
            precautions = """
            - Reduce sugar intake
            - Exercise regularly
            - Maintain healthy weight
            - Monitor glucose level
            - Eat more vegetables and fiber-rich food
            - Go for routine health checkups
            """
            st.warning(f"⚠️ Doctor Recommendation: {doctor_advice}")
            st.markdown("### 🛡️ Precautions")
            st.markdown(precautions)

        else:
            doctor_advice = "Current risk is low, but regular health checkups are still recommended."
            precautions = """
            - Maintain a healthy lifestyle
            - Eat a balanced diet
            - Stay physically active
            - Avoid excessive sugar
            - Sleep properly
            - Go for regular checkups
            """
            st.success(f"✅ Doctor Recommendation: {doctor_advice}")
            st.markdown("### 🛡️ Prevention Tips")
            st.markdown(precautions)

        st.session_state.last_prediction = {
            "result": result_text,
            "risk_level": risk_level,
            "probability": diabetes_probability,
            "glucose": glucose,
            "bmi": bmi,
            "age": age
        }

        history_file = "patient_history.csv"

        new_record = pd.DataFrame([{
            "Patient Name": patient_name,
            "Pregnancies": preg,
            "Glucose": glucose,
            "Blood Pressure": bp,
            "Skin Thickness": skin,
            "Insulin": insulin,
            "BMI": bmi,
            "Diabetes Pedigree Function": dpf,
            "Age": age,
            "Prediction Result": result_text,
            "Risk Level": risk_level,
            "Probability": round(diabetes_probability, 2),
            "Doctor Advice": doctor_advice
        }])

        if os.path.exists(history_file):
            old_history = pd.read_csv(history_file)
            updated_history = pd.concat([old_history, new_record], ignore_index=True)
        else:
            updated_history = new_record

        updated_history.to_csv(history_file, index=False)

        report = f"""
DIABETES PREDICTION REPORT

Patient Name: {patient_name}

Pregnancies: {preg}
Glucose Level: {glucose}
Blood Pressure: {bp}
Skin Thickness: {skin}
Insulin: {insulin}
BMI: {bmi}
Diabetes Pedigree Function: {dpf}
Age: {age}

Prediction Result: {result_text}
Risk Level: {risk_level}
Diabetes Risk Probability: {diabetes_probability:.2f}%

Doctor Advice:
{doctor_advice}

Precautions:
{precautions}

Note: This is an educational ML project, not a medical diagnosis.
"""

        st.download_button(
            label="📄 Download Prediction Report",
            data=report,
            file_name="diabetes_prediction_report.txt",
            mime="text/plain"
        )

        st.success("✅ Patient prediction saved successfully.")
        st.markdown("</div>", unsafe_allow_html=True)


with tab2:
    st.markdown("<div class='premium-card'>", unsafe_allow_html=True)
    st.subheader("📊 Dataset Preview")
    st.dataframe(df.head(), use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    graph1 = px.histogram(df, x="Glucose", color="Outcome", title="Glucose Level Distribution")
    st.plotly_chart(graph1, use_container_width=True)

    graph2 = px.scatter(
        df,
        x="BMI",
        y="Glucose",
        color="Outcome",
        size="Age",
        title="BMI vs Glucose with Age"
    )
    st.plotly_chart(graph2, use_container_width=True)

    graph3 = px.box(df, x="Outcome", y="Age", color="Outcome", title="Age Distribution by Diabetes Outcome")
    st.plotly_chart(graph3, use_container_width=True)

    corr = df.corr()
    graph4 = px.imshow(corr, text_auto=True, title="Feature Correlation Heatmap")
    st.plotly_chart(graph4, use_container_width=True)

    importance = model.feature_importances_
    features = X.columns

    graph5 = px.bar(
        x=features,
        y=importance,
        title="Feature Importance",
        labels={"x": "Features", "y": "Importance"}
    )
    st.plotly_chart(graph5, use_container_width=True)


with tab3:
    st.markdown("<div class='premium-card'>", unsafe_allow_html=True)
    st.subheader("🧪 Model Accuracy Comparison")

    comparison_df = pd.DataFrame({
        "Model": list(model_scores.keys()),
        "Accuracy": [round(score * 100, 2) for score in model_scores.values()]
    })

    st.dataframe(comparison_df, use_container_width=True)

    comparison_graph = px.bar(
        comparison_df,
        x="Model",
        y="Accuracy",
        title="Model Accuracy Comparison",
        text="Accuracy"
    )
    st.plotly_chart(comparison_graph, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='premium-card'>", unsafe_allow_html=True)
    st.subheader("🧩 Confusion Matrix - Random Forest")

    cm = confusion_matrix(y_test, y_pred)

    cm_df = pd.DataFrame(
        cm,
        index=["Actual No Diabetes", "Actual Diabetes"],
        columns=["Predicted No Diabetes", "Predicted Diabetes"]
    )

    st.dataframe(cm_df, use_container_width=True)

    cm_graph = px.imshow(
        cm,
        text_auto=True,
        labels=dict(x="Predicted", y="Actual", color="Count"),
        x=["No Diabetes", "Diabetes"],
        y=["No Diabetes", "Diabetes"],
        title="Confusion Matrix Heatmap"
    )
    st.plotly_chart(cm_graph, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)


with tab4:
    st.markdown("<div class='premium-card'>", unsafe_allow_html=True)
    st.subheader("🗂️ Patient Prediction History")

    history_file = "patient_history.csv"

    if os.path.exists(history_file):
        history_df = pd.read_csv(history_file)
        st.dataframe(history_df, use_container_width=True)

        csv_data = history_df.to_csv(index=False)

        st.download_button(
            label="⬇️ Download Patient History CSV",
            data=csv_data,
            file_name="patient_history.csv",
            mime="text/csv"
        )

        if st.button("🗑️ Clear Patient History"):
            os.remove(history_file)
            st.success("Patient history cleared. Refresh the app.")
    else:
        st.info("No patient history yet. Make a prediction first.")

    st.markdown("</div>", unsafe_allow_html=True)


st.markdown("""
<div class="disclaimer">
⚠️ Disclaimer: This app is only for educational purposes. It is not a medical diagnosis.
Please consult a doctor for real health advice.
</div>
""", unsafe_allow_html=True)