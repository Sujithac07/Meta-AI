import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import streamlit as st
import pandas as pd
from mlops.mlflow_utils import (
    start_experiment,
    log_params,
    log_decision,
    end_experiment
)


if "analysis_done" not in st.session_state:
    st.session_state.analysis_done = False

if "final_decision" not in st.session_state:
    st.session_state.final_decision = None

if "risk_out" not in st.session_state:
    st.session_state.risk_out = None

from chatbot.google_adk_agent import GoogleADKMetaAgent
from agents.memory_agent import MemoryAgent


st.set_page_config(page_title="Meta AI: An Intelligent System for Automated Machine Learning Pipeline Generation", layout="wide")

st.title("Meta AI: An Intelligent System for Automated Machine Learning Pipeline Generation")
st.subheader("Agentic, Self-Reasoning ML System")

st.write(
    "Upload a dataset and let the Meta AI system reason, audit, and decide "
    "whether a machine learning model can be trusted."
)

# ---------------- DATASET UPLOAD ----------------
uploaded_file = st.file_uploader(
    "Upload CSV Dataset", type=["csv"]
)

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.success("Dataset uploaded successfully")
    st.write("Preview of dataset:")
    st.dataframe(df.head())

    # Target selection
    target_col = st.selectbox(
        "Select Target Column", df.columns
    )

    # ---------------- RUN META-AI ANALYSIS ----------------

    from pipeline.meta_ai_pipeline import run_meta_ai_pipeline
    if st.button("Run Meta AI Analysis"):
        st.session_state.pipeline_out = run_meta_ai_pipeline(df, target_col)
        st.session_state.analysis_done = True

    # ---------------- DISPLAY RESULTS ----------------

    if "pipeline_out" in st.session_state and st.session_state.analysis_done:
        out = st.session_state.pipeline_out

        st.subheader("Final Decision")
        st.write(out["final_decision"]["final_decision"])
        st.write(out["final_decision"]["reason"])

        st.subheader("Data Profiling")
        st.write(out["data"])

        st.subheader("Risk Analysis")
        st.write(out["risk"])

        st.subheader("Strategy Selection")
        st.write(out["strategy"])

        st.subheader("Model Metrics")
        st.write("All Model Results:")
        for model_name, result in out["metrics"].items():
            if isinstance(result, dict) and 'metrics' in result:
                st.write(f"**{model_name}:** {result['metrics']}")
            else:
                st.write(f"**{model_name}:** {result}")
        st.write("**Best Model Selected:** Based on primary metric.")

        st.subheader("Feature Importance")
        st.write(out["feature_importance"])

        st.subheader("Stability Analysis")
        st.write(out.get("stability", {}))

        st.subheader("Data Validation")
        st.write(out.get("validation", {}))

        st.subheader("Model Card")
        st.write(out.get("model_card", {}))

        st.subheader("Report")
        st.write(out.get("report_path", "Not generated"))

        st.subheader("Failure Analysis")
        st.write(out.get("failure_analysis", {}))

        st.subheader("LLM Insights")
        llm_insights = out.get("llm_insights")
        if llm_insights:
            st.write(llm_insights)
        else:
            st.info("Set OPENAI_API_KEY to enable LLM insights.")

        # ---------- MODEL VERSIONING (MLOPS) ----------
        if out["final_decision"]["final_decision"] in ["ACCEPT", "ACCEPT_WITH_WARNINGS"]:
            # The model is already registered in the pipeline
            pass

        # ================= GOOGLE ADK EXPLANATION AGENT =================

        st.subheader("Meta AI Explanation Agent (Google ADK)")

        try:
            adk_agent = GoogleADKMetaAgent()
        except Exception as e:
            adk_agent = None
            st.warning(f"Explanation agent unavailable: {e}")

        memory_agent = MemoryAgent()

        user_question = st.text_input(
            "Ask Meta AI (why, risks, safety, future, trust...)"
        )

        if user_question:
            memory_context = memory_agent.consult(user_question)

            system_state = {
                "final_decision": out["final_decision"]["final_decision"],
                "reason": out["final_decision"]["reason"],
                "agent_trace": out["final_decision"]["trace"],
                "risks": out["risk"]["risks"]
            }

            if adk_agent:
                answer = adk_agent.answer(
                    user_question,
                    system_state,
                    memory_context
                )
                st.success(answer)
            else:
                st.info("Set GOOGLE_API_KEY or GEMINI_API_KEY to enable the explanation agent.")

            # ---------------- MLOPS TRACKING ----------------

            start_experiment()

            log_params({
                "target_column": target_col,
                "dataset_rows": df.shape[0],
                "dataset_columns": df.shape[1]
            })

            log_params({
                "overfitting_risk": out["risk"]["risks"]["overfitting_risk"],
                "class_imbalance": out["risk"]["risks"]["class_imbalance"]
            })

            log_decision(out["final_decision"]["final_decision"])

            end_experiment()
