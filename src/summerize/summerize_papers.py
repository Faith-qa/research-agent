import json
from pathlib import Path
from typing import List, Dict
from loguru import logger
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from src.utils.config_loader import load_config


class PaperSummarizer:
    def __init__(self):
        self.config = load_config()
        self.llm = ChatOpenAI(
            model=self.config["llm"]["model"],
            temperature=self.config["llm"]["temperature"]
        )
        self.prompt = PromptTemplate(
            input_variables=["abstract"],
            template=(
                "You are an AI research assistant. Summarize the following abstract in 2-3 sentences, "
                "focusing on the main contribution and findings. Then, extract 1-2 specific open problems "
                "or research challenges mentioned or implied.\n\n"
                "Abstract: {abstract}\n\n"
                "Output format (JSON):\n"
                "{\n"
                '  "summary": "<summary text>",\n'
                '  "open_problems": ["<problem 1>", "<problem 2>"]\n'
                "}"
            )
        )
        self.chain = LLMChain(llm=self.llm, prompt=self.prompt)

    def summarize(self, abstract: str) -> Dict:
        """Summarize a single paper abstract."""
        result = self.chain.run(abstract=abstract)
        return json.loads(result)

    def summarize_papers(self, input_file: str = "data/papers.json", output_file: str = "data/summaries.json") -> List[Dict]:
        """Summarize all papers from a JSON file and save the results."""
        input_path = Path(input_file)
        if not input_path.exists():
            logger.error(f"Input file {input_file} not found.")
            return []

        with input_path.open("r") as f:
            papers = json.load(f)

        summaries = []
        for paper in papers:
            try:
                result_json = self.summarize(paper["abstract"])
                summaries.append({
                    "title": paper["title"],
                    "id": paper["id"],
                    "source": paper["source"],
                    "summary": result_json["summary"],
                    "open_problems": result_json["open_problems"]
                })
                logger.info(f"Summarized paper: {paper['title']}")
            except Exception as e:
                logger.exception(f"Error summarizing paper '{paper['title']}': {e}")

        # Save summaries
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w") as f:
            json.dump(summaries, f, indent=2)

        logger.info(f"Saved {len(summaries)} summaries to {output_file}")
        return summaries


