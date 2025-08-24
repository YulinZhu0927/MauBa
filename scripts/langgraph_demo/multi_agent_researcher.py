from typing import TypedDict, List


class ResearchState(TypedDict):
    task: dict
    initial_research: str
    sections: List[str]
    research_data: List[dict]
    # Report layout
    title: str
    headers: dict
    date: str
    table_of_contents: str
    introduction: str
    conclusion: str
    sources: List[str]
    report: str


from langgraph.graph import StateGraph
workflow = StateGraph(ResearchState)

from gpt_researcher import GPTResearcher

class ResearchAgent:
    def __init__(self):
        pass

    async def research(self, query: str):
        # Initialize the researcher
        researcher = GPTResearcher(parent_query=parent_query, query=query, report_type=research_report, config_path=None)
        # Conduct research on the given query
        await researcher.conduct_research()
        # Write the report
        report = await researcher.write_report()

        return report
