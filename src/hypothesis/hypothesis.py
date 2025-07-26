import json
from pathlib import Path
from typing import List, Dict
from loguru import logger
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import SequentialChain, LLMChain
from src.utils.config_loader import load_config


class HypothesisGenerator:
    """
    HypothesisGenerator Orchestrates the process of analyzing open research problems,
    detecting trends, and generating experiment hypotheses using LLMS (via LangChain)
    """

    # Prompt template for analyzing trends from open problems
    ANALYSIS_TEMPLATE = """
    Analyze the following open research problems and identify key trends or common themes.
    Problems: {problems}
    
    Output format (JSON):
    {{
        "trends": ["<trend 1>", "<trend 2>"]
    }}
    """

    # Prompt template for generating hypothesis from open trends and open problems
    HYPOTHESIS_TEMPLATE = """
    Based on the following open problems and trends, propose 3 novel experiment ideas.
    Each idea should include:
    - Hypothesis: A clear, testable statement.
    - Setup: A brief description of the experiment.
    - Expected Outcomes: Potential impact or findings.
    
    Problems: {problems}
    Trends: {trends}
    
    Output format (JSON):
    [
        {{
            "hypothesis": "<hypothesis>",
            "setup": "<setup>",
            "expected_outcomes": "<outcomes>"
        }},
        ...
    ]
    """
    def __int__(self):
        """
        Initializing the hypothesis generator by:
        - Loading config.yaml file
        - initializing the LLM (ChatOpenAI)
        - Setting up the analysis and hypothesis chains
        """
        self.config = load_config()
        self.llm = ChatOpenAI(
            model = self.config["llm"]["model"],
            temperature=self.config.get("llm", {}).get("temperature", 0.7)
        )

        # Chain 1: Analyze open problems to identify trends
        self.analysis_chain = LLMChain(
            llm=self.llm,
            prompt = PromptTemplate(input_variables=["problems"], template=self.ANALYSIS_TEMPLATE),
            output_key="trends"
        )

        # Chain 2: Generate hypothesis based on problems and trends
        self.hypothesis_chain = LLMChain(
            llm=self.llm,
            prompt = PromptTemplate(input_variables=["problems", "trends"], template=self.HYPOTHESIS_TEMPLATE),
            output_key="hypothesis"
        )

        # SequentialChain runs analisyis first: then uses it's output for hypothesis generation
        self.overall_chain = SequentialChain(
            chains = [self.analysis_chain, self.hypothesis_chain],
            inpuut_variables=["problems"],
            output_variables = ["hypothesis"]
        )

    def _load_open_problems(self, file_path:str = "data/summaries.json")-> List[str]:
        path = Path(file_path)
        if not path.exists():
            logger.error(f"summeries file not found:{file_path}")
            return []
        with path.open("r") as f:
            summeries = json.load(f)
        return [problem for summary in summeries for problem in summeries.get("open_problems", [])]

    def generate(self, input_file: str = "data/summaries.json", output_file: str = "data/hypotheses.json")->List[Dict]:
        problems = self._load_open_problems(input_file)
        if not problems:
            logger.warning("No open problems found for hypothesis generation.")
            return []

        logger.info(f"generating hypothesis from {len(problems)} open problems.")
        try:
            result = self.overall_chain({"problems": "\n".join(problems)})
            hypotheses = json.loads(result["hypothesis"])
        except Exception as e:
            logger.exception(f"Failed to generate hypothesis:{e}")
            return []

        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w") as f:
            json.dump(hypotheses, f, indent=2)

        logger.info(f"Generated  {len(hypotheses)} hypotheses and saved to {output_file}")

        return hypotheses



