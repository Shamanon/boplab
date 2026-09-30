from langchain_community.chat_models import ChatOllama
from langgraph.graph import END, StateGraph
from typing_extensions import TypedDict


# 1. Point LangChain directly to your K3s ollama-partner endpoint
llm = ChatOllama(
    base_url="http://localhost:31434",  # or internal cluster URL
    model="deepseek-r1:latest",  # Use DeepSeek-R1 or Qwen2.5-Coder on gpu1!
    temperature=0.2,
)


# 2. Define the Agent State
class ResearchState(TypedDict):
  topic: str
  queries: list[str]
  research_notes: list[str]
  final_report: str
  iteration: int


# Tomorrow we'll wire this into LangSmith tracing:
# export LANGCHAIN_TRACING_V2="true"
# export LANGCHAIN_API_KEY="your-langsmith-key"
