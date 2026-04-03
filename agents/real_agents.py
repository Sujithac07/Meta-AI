import os

# ==========================================
# ROBUST IMPORT SYSTEM
# ==========================================

# Pre-define names to avoid NameError if imports fail
# We use a global registry to ensure visibility
GLOBALS = globals()
GLOBALS['ChatOpenAI'] = None
GLOBALS['Agent'] = None
GLOBALS['Task'] = None
GLOBALS['Crew'] = None
GLOBALS['Process'] = None

try:
    from crewai import Agent, Task, Crew, Process
    from langchain_openai import ChatOpenAI
    AGENTS_AVAILABLE = True
    # Re-verify they are actually bound
    GLOBALS['ChatOpenAI'] = ChatOpenAI
    GLOBALS['Agent'] = Agent
    GLOBALS['Task'] = Task
    GLOBALS['Crew'] = Crew
    GLOBALS['Process'] = Process
except Exception as e:
    AGENTS_AVAILABLE = False
    print(f"⚠️ [real_agents] Agentic components disabled (Critical Error): {e}")

# ==========================================
# AGENT DELEGATION
# ==========================================

class MetaAIAgents:
    def __init__(self):
        self.llm = None
        if AGENTS_AVAILABLE and GLOBALS.get('ChatOpenAI'):
            api_key = os.getenv("OPENAI_API_KEY")
            if api_key:
                try:
                    self.llm = GLOBALS['ChatOpenAI'](
                        model="gpt-4o",
                        temperature=0.2,
                        api_key=api_key 
                    )
                except Exception as e:
                    print(f"❌ [MetaAIAgents] LLM Init Failed: {e}")
            else:
                print("⚠️ [MetaAIAgents] No API Key Found.")

    def _create_agent(self, role, goal, backstory):
        if not AGENTS_AVAILABLE or not self.llm:
            return None
        try:
            return GLOBALS['Agent'](
                role=role,
                goal=goal,
                backstory=backstory,
                verbose=True,
                allow_delegation=False,
                llm=self.llm
            )
        except Exception:
            return None

    def data_scientist_agent(self):
        return self._create_agent(
            'Senior Data Scientist',
            'Analyze data patterns, quality, and statistical properties deeply.',
            "Veteran Data Scientist. Expert in spotting data quality issues and anomalies."
        )

    def risk_analyst_agent(self):
        return self._create_agent(
            'AI Risk Officer',
            'Evaluate ML solutions for bias, instability, and compliance risks.',
            "Strict Risk Officer. Audits strategies for safety and robustness."
        )

    def ml_engineer_agent(self):
        return self._create_agent(
            'Lead ML Engineer',
            'Design and implement optimal ML training strategies.',
            "Pragmatic ML Engineer. Expert in architectures (XGBoost, TabNet) and tuning."
        )

    def debugger_agent(self):
        return self._create_agent(
            'Autonomous Model Debugger',
            'Identify flaws in model performance and suggest code/logic fixes.',
            "Post-mortem specialist. Finds exactly WHY a model underperforms."
        )

    def feature_architect_agent(self):
        return self._create_agent(
            'Feature Architect',
            'Perform advanced feature engineering and transformation logic.',
            "Wizard of feature engineering. Expert in interaction terms and encodings."
        )

class MetaAITasks:
    def profile_data_task(self, agent, data_summary):
        if not AGENTS_AVAILABLE or not agent:
            return None
        return GLOBALS['Task'](
            description=f"Analyze data profile: {data_summary}. Generate technical Data Audit Report.",
            agent=agent,
            expected_output="Detailed markdown report of data quality."
        )

    def assess_risk_task(self, agent, data_audit):
        if not AGENTS_AVAILABLE or not agent:
            return None
        return GLOBALS['Task'](
            description=f"Review audit: {data_audit}. Produce Risk Assessment Matrix.",
            agent=agent,
            expected_output="Risk matrix with ratings and mitigations."
        )

    def design_strategy_task(self, agent, risk_assessment):
        if not AGENTS_AVAILABLE or not agent:
            return None
        return GLOBALS['Task'](
            description=f"Based on risk {risk_assessment}, design technical Training Strategy (JSON).",
            agent=agent,
            expected_output="Technical strategy doc with JSON config."
        )

    def critique_strategy_task(self, agent, strategy_doc):
        if not AGENTS_AVAILABLE or not agent:
            return None
        return GLOBALS['Task'](
            description=(
                "CRITICAL REVIEW REQUIRED. "
                f"Review the proposed Training Strategy: {strategy_doc}. "
                "Does it adequately ignore the bias issues you found? "
                "Are the selected models too complex or opaque? IS IT SAFE? "
                "Output a critique report listing specific weaknesses and demands for change."
            ),
            agent=agent,
            expected_output="Critique report highlighting weaknesses."
        )

    def consensus_task(self, agent, critique):
        if not AGENTS_AVAILABLE or not agent:
            return None
        return GLOBALS['Task'](
            description=(
                f"Review the Critique: {critique}. Finalize the strategy. "
                "Accept valid criticism where needed and Produce the FINAL binding strategy."
            ),
            agent=agent,
            expected_output="The final, approved strategy document."
        )

    def feature_engineering_task(self, agent, data_audit):
        if not AGENTS_AVAILABLE or not agent:
            return None
        return GLOBALS['Task'](
            description=f"Analyze audit {data_audit}. Propose 5 advanced feature transformations.",
            agent=agent,
            expected_output="Prioritized list of transformations."
        )

    def debug_metrics_task(self, agent, metrics_summary):
        if not AGENTS_AVAILABLE or not agent:
            return None
        return GLOBALS['Task'](
            description=f"Analyze metrics {metrics_summary}. Suggest 3 specific hyperparameter fixes.",
            agent=agent,
            expected_output="Technical diagnostic report with fixes."
        )

def run_meta_ai_crew(data_profile_str):
    if not AGENTS_AVAILABLE:
        return "⚠️ Agent Council Unavailable: Missing dependencies or DLL errors. Reverting to heuristic logic."

    try:
        agents = MetaAIAgents()
        if not agents.llm:
            return "⚠️ Agent Council Offline: No API Key or LLM init failed."

        tasks = MetaAITasks()
        
        # Instantiate agents
        ds = agents.data_scientist_agent()
        ro = agents.risk_analyst_agent()
        mle = agents.ml_engineer_agent()
        fa = agents.feature_architect_agent()

        if not all([ds, ro, mle, fa]):
            return "⚠️ Task instantiation failed: Agents could not be initialized."

        # Define tasks
        t1 = tasks.profile_data_task(ds, data_profile_str)
        t_feat = tasks.feature_engineering_task(fa, t1)
        t2 = tasks.assess_risk_task(ro, t_feat) 
        t3 = tasks.design_strategy_task(mle, t2)
        t4 = tasks.critique_strategy_task(ro, t3) 
        t5 = tasks.consensus_task(mle, t4)           

        crew = GLOBALS['Crew'](
            agents=[ds, ro, mle, fa],
            tasks=[t1, t_feat, t2, t3, t4, t5],
            verbose=True,
            process=GLOBALS['Process'].sequential,
        )

        result = crew.kickoff()
        return f"### 🤖 Agentic Consensus Achieved\n\n{result}"

    except Exception as e:
        return f"⚠️ Agent Workflow Failed: {str(e)}. Falling back to baseline heuristics."
