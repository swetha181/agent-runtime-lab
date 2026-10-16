import pytest

from agent_lab import build_context
from agent_lab.agent import Agent
from agent_lab.config import Settings
from agent_lab.llm import ScriptedLLM
from agent_lab.router import RuleRouter


@pytest.fixture
def ctx():
    return build_context(Settings(offline=True))


@pytest.fixture
def make_agent(ctx):
    """Agent with scripted LLM responses and the deterministic router."""

    def _make(responses, router=None):
        llm = ScriptedLLM(responses)
        return Agent(llm, ctx, router=router or RuleRouter()), llm

    return _make
