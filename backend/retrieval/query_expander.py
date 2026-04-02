"""Query expansion using LLM-generated synonyms.

Expands user queries with contextually appropriate alternative terms
to improve retrieval of documents that use different terminology.
Works across all scientific domains by using an LLM to generate
domain-appropriate synonyms at query time.
"""

import logging
from typing import List, Tuple

from anthropic import Anthropic

logger = logging.getLogger(__name__)


EXPANSION_PROMPT = """Given this scientific research query, suggest 3-5 alternative terms or phrasings that academic papers might use for the same concepts. Focus on:
- Synonyms and abbreviations (e.g., "heart attack" → "myocardial infarction", "MI")
- Full forms of acronyms or vice versa
- Related technical terminology that papers in this area commonly use

Query: {query}

Return ONLY the alternative terms, comma-separated, on a single line. Do not include explanations."""


class QueryExpander:
    """Expand queries with LLM-generated domain synonyms."""

    def __init__(
        self,
        anthropic_client: Anthropic,
        model: str = "claude-haiku-4-5-20251001",
        max_expansion_terms: int = 5,
    ):
        """Initialize query expander.

        Args:
            anthropic_client: Anthropic API client
            model: Model to use for expansion (should be fast/cheap)
            max_expansion_terms: Maximum number of synonym terms to add
        """
        self.client = anthropic_client
        self.model = model
        self.max_expansion_terms = max_expansion_terms

    def expand_query(self, query: str) -> Tuple[str, List[str]]:
        """Expand query with LLM-generated synonyms.

        Args:
            query: Original user query

        Returns:
            Tuple of (expanded_query, list_of_added_terms)
        """
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=150,
                temperature=0.3,
                messages=[{
                    "role": "user",
                    "content": EXPANSION_PROMPT.format(query=query),
                }],
            )

            text = response.content[0].text.strip()
            # Parse comma-separated terms
            terms = [t.strip().strip('"\'') for t in text.split(",")]
            # Filter empty and overly long terms, limit count
            terms = [t for t in terms if t and len(t) < 80]
            added_terms = terms[:self.max_expansion_terms]

            if added_terms:
                expanded = f"{query} (related terms: {', '.join(added_terms)})"
                logger.debug(f"LLM expanded query: added {added_terms}")
                return expanded, added_terms

        except Exception as e:
            logger.warning(f"LLM query expansion failed: {e}")

        return query, []
