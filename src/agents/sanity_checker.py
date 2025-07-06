import json
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage

from typing_extensions import Literal
from pydantic import BaseModel, Field
from src.data.models import FinancialMetrics
from src.graph.state import AgentState, show_agent_reasoning
from src.tools.api import get_financial_metrics
from src.utils.llm import call_llm
from src.utils.progress import progress

class ValidityDecision(BaseModel):
    validity: Literal["valid", "invalid"]
    reasoning: str = Field(description="Reasoning for the decision")

def sanity_checker_agent(state: AgentState):
    """
    Checks the responses from each of the analysts to make sure their reasoning is valid.
    1. Ensures that the reasoning is related to finances
    2. Ensures that the reasoning is related to the corresponding ticker
    3. Ensures that the signal matches the reasoning
    """
    analyst_signals = state["data"]["analyst_signals"]

    sanity_checks: dict[str, dict] = {}
    for agent, signals in analyst_signals.items():
        if agent == "risk_management_agent":
            continue

        progress.update_status("sanity_checker", agent, "Checking analyst")
        sanity_check = generate_sanity_check(
            analyst=agent,
            analyses=signals,
            state=state
        )
        sanity_checks[agent] = {
            "valid": sanity_check.validity == "valid",
            "reasoning": sanity_check.reasoning
        }

        progress.update_status("sanity_checker", agent, "Done", analysis=sanity_check.reasoning)
    
    # Create the message
    message = HumanMessage(content=json.dumps(sanity_checks), name="sanity_checker_agent")

    # Show reasoning if requested
    if state["metadata"]["show_reasoning"]:
        show_agent_reasoning(sanity_checks, "Sanity Checker")
    
    # Get rid of invalid agents
    for agent in {agent for agent, analysis in sanity_checks.items() if not analysis["valid"]}:
        analyst_signals.pop(agent)

    progress.update_status("sanity_checker", None, "Done")

    return {"messages": [message], "data": state["data"]}

def generate_sanity_check(
        analyst: str,
        analyses: dict,
        state: AgentState
) -> ValidityDecision:
    # Create the prompt template
    template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
                You are a final sanity check agent that verifies analysts on various tickers.
                ### Objectives:
                - Ensure that the agent's "reasoning" for each ticker is related to finances.
                - Ensure that the agent's "reasoning" matches the corresponding ticker.
                - Ensure that the agent's "signal" makes sense given the corresponding "reasoning" for each ticker.
                These are all **necessary** conditions that must be met.

                ### Available Actions
                - "valid": all of the above conditions are met
                - "invalid": any of the above conditions are violated
                """
            ),
            (
                "human",
                """
                analyst: {analyst}

                analyses:
                {analyses}

                Output strictly in JSON with the following structure
                {{
                    "validity": "valid/invalid",
                    "reasoning": string
                }}
                """
            )
        ]
    )
    
    # Generate the prompt
    prompt = template.invoke(
        {
            "analyst": analyst,
            "analyses": json.dumps(analyses, indent=2)
        }
    )

    # Create the default facotry for SanityCheckerOutput
    def create_default_sanity_check_output():
        return ValidityDecision(validity="invalid", reasoning="Error in sanity check. Defaulting to invalid")
    
    # Make the LLM call
    return call_llm(
        prompt=prompt,
        pydantic_model=ValidityDecision,
        agent_name="sanity_checker",
        state=state,
        default_factory=create_default_sanity_check_output
    ) # type: ignore
    