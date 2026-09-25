import json
import requests
from ddgs import DDGS
from crewai.tools import tool


@tool("Search scholarly publication records")
def search_papers(query: str) -> str:
    """Search Crossref for publication metadata and DOI links; this does not read full papers."""
    try:
        response = requests.get(
            "https://api.crossref.org/works",
            params={"query": query, "rows": 7, "select": "DOI,title,author,published,container-title,abstract,URL,type"},
            headers={"User-Agent": "AIResearchAgent/0.1 (academic prototype)"},
            timeout=20,
        )
        response.raise_for_status()
        papers = []
        for item in response.json()["message"]["items"]:
            parts = item.get("published", {}).get("date-parts", [[]])[0]
            authors = item.get("author", [])[:4]
            papers.append({
                "title": (item.get("title") or [""])[0],
                "year": parts[0] if parts else None,
                "authors": ", ".join(" ".join(filter(None, [a.get("given"), a.get("family")])) for a in authors),
                "doi": item.get("DOI"),
                "url": item.get("URL"),
                "journal": (item.get("container-title") or [""])[0],
                "abstract_available": bool(item.get("abstract")),
                "abstract": item.get("abstract", "")[:1800],
            })
        return json.dumps(papers, ensure_ascii=False)
    except (requests.RequestException, KeyError, ValueError) as exc:
        return f"Crossref search failed: {exc}"


@tool("Search the web")
def search_web(query: str) -> str:
    """Find web pages relevant to a research topic; snippets do not prove study findings."""
    try:
        results = DDGS(timeout=10).text(query, max_results=5)
        return json.dumps([{"title": r.get("title"), "url": r.get("href"), "snippet": r.get("body")} for r in results], ensure_ascii=False)
    except Exception as exc:
        return f"Web search failed: {exc}"
