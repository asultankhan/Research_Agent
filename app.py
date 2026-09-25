import os
import streamlit as st
from crewai import Agent, Crew, LLM, Process, Task
from research_tools import search_web, search_papers

st.set_page_config(page_title="AI Research Agent", page_icon="🔎")
st.title("AI Research Agent")
st.caption("A single CrewAI agent. Review the linked sources before using its findings in a publication.")

# Root-level Streamlit secrets are also exposed as environment variables.
try:
    groq_key = st.secrets.get("GROQ_API_KEY", os.getenv("GROQ_API_KEY", ""))
except FileNotFoundError:
    groq_key = os.getenv("GROQ_API_KEY", "")

if not groq_key:
    st.error("Set GROQ_API_KEY in Streamlit Secrets to use this app.")
    st.stop()
os.environ["GROQ_API_KEY"] = groq_key

with st.form("research"):
    question = st.text_area("Your research question", placeholder="What does research say about AI tutors in online higher education?")
    submitted = st.form_submit_button("Research")

if submitted:
    if not question.strip():
        st.warning("Enter a research question first.")
    else:
        with st.spinner("Searching and preparing your brief. This may take a minute..."):
            try:
                llm = LLM(model="groq/openai/gpt-oss-120b", temperature=0.2)
                researcher = Agent(
                    role="Academic research assistant",
                    goal="Find traceable academic sources and write cautious research briefs",
                    backstory="You distinguish search results and metadata from actual full-text evidence.",
                    tools=[search_papers, search_web],
                    llm=llm,
                    allow_delegation=False,
                    max_iter=8,
                    verbose=False,
                )
                task = Task(
                    description=(
                        f"Investigate this question: {question.strip()}\n"
                        "Use the scholarly publication tool first; optionally use web search for context. "
                        "Include up to five relevant papers with title, year, DOI link, and why relevant. "
                        "Then give a short synthesis and possible gaps. "
                        "Explicitly state whether your observations come from metadata, an abstract, or a full text. "
                        "Do not invent findings, methods, sample sizes, quotations, or citations. "
                        "If source information is insufficient, say so."
                    ),
                    expected_output="A concise Markdown research brief with a source table, synthesis, gaps, and evidence limitations.",
                    agent=researcher,
                )
                result = Crew(agents=[researcher], tasks=[task], process=Process.sequential, verbose=False).kickoff()
                report = result.raw
                st.markdown(report)
                st.download_button("Download brief", report, file_name="research_brief.md", mime="text/markdown")
            except Exception as exc:
                st.error(f"Research failed: {exc}")
