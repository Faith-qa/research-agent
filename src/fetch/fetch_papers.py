import arxiv
from semanticscholar import SemanticScholar
import json
import os
import yaml
from loguru import logger
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict


class FetchPapers:
    def __init__(self, query: str, max_results: int):
        self.query = query
        self.max_results = max_results
        self.config = self._load_config()
        self.arxiv_client = arxiv.Client()
        self.semantic_client = SemanticScholar()

    def _load_config(self) -> Dict:
        with open("config.yaml", "r") as f:
            return yaml.safe_load(f)

    def _save_papers(self, papers: List[Dict], filename: str = "data/papers.json") -> None:
        os.makedirs("data", exist_ok=True)
        with open(filename, "w") as f:
            json.dump(papers, f, indent=2)

    def fetch_arxiv_papers(self, max_results: int) -> List[Dict]:
        search = arxiv.Search(query=self.query, max_results=max_results, sort_by=arxiv.SortCriterion.SubmittedDate)
        return [
            {
                "title": result.title,
                "abstract": result.summary,
                "id": result.entry_id,
                "published": str(result.published),
                "authors": [author.name for author in result.authors],
                "source": "arXiv"
            }
            for result in self.arxiv_client.results(search)
        ]

    def fetch_semantic_scholar_papers(self, max_results: int) -> List[Dict]:
        results = self.semantic_client.search_paper(self.query, limit=max_results)
        return [
            {
                "title": paper.title,
                "abstract": paper.abstract or "No abstract available",
                "id": paper.paperId,
                "published": paper.publicationDate or "unknown",
                "authors": [author["name"] for author in paper.authors],
                "source": "Semantic Scholar"
            }
            for paper in results
        ]

    def fetch_papers(self) -> List[Dict]:
        half = self.max_results // 2
        logger.info(f"Fetching {self.max_results} papers (arXiv & Semantic Scholar)...")

        with ThreadPoolExecutor() as executor:
            future_arxiv = executor.submit(self.fetch_arxiv_papers, half)
            future_ss = executor.submit(self.fetch_semantic_scholar_papers, half)
            arxiv_papers = future_arxiv.result()
            ss_papers = future_ss.result()

        papers = arxiv_papers + ss_papers
        self._save_papers(papers)
        logger.info(f"Fetched {len(papers)} papers.")
        return papers
