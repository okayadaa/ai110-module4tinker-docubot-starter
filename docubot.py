"""
Core DocuBot class responsible for:
- Loading documents from the docs/ folder
- Building a simple retrieval index (Phase 1)
- Retrieving relevant snippets (Phase 1)
- Supporting retrieval only answers
- Supporting RAG answers when paired with Gemini (Phase 2)
"""

import os
import glob
import string


NO_INFO_RESPONSE = "I'm unsure. I can't provide any info"


STOPWORDS = {
    "a",
    "an",
    "the",
    "and",
    "or",
    "but",
    "if",
    "then",
    "else",
    "to",
    "of",
    "in",
    "on",
    "for",
    "with",
    "at",
    "by",
    "from",
    "as",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "do",
    "does",
    "did",
    "what",
    "whats",
    "who",
    "when",
    "where",
    "why",
    "how",
    "can",
    "could",
    "should",
    "would",
    "will",
    "this",
    "that",
    "these",
    "those",
    "i",
    "you",
    "we",
    "they",
    "it",
    "my",
    "your",
    "our",
    "their",
}

class DocuBot:
    def __init__(self, docs_folder="docs", llm_client=None):
        """
        docs_folder: directory containing project documentation files
        llm_client: optional Gemini client for LLM based answers
        """
        self.docs_folder = docs_folder
        self.llm_client = llm_client

        # Load documents into memory
        self.documents = self.load_documents()  # List of (filename, text)

        # Build a retrieval index (implemented in Phase 1)
        self.index = self.build_index(self.documents)

    def _tokenize(self, text, remove_stopwords=False):
        """
        Lowercase and tokenize text by whitespace with punctuation stripped.
        Optionally remove stopwords to reduce weak keyword matches.
        """
        tokens = []
        for raw_word in text.lower().split():
            token = raw_word.strip(string.punctuation)
            if not token:
                continue
            if remove_stopwords and token in STOPWORDS:
                continue
            tokens.append(token)
        return tokens

    # -----------------------------------------------------------
    # Document Loading
    # -----------------------------------------------------------

    def load_documents(self):
        """
        Loads all .md and .txt files inside docs_folder.
        Returns a list of tuples: (filename, text)
        """
        docs = []
        pattern = os.path.join(self.docs_folder, "*.*")
        for path in glob.glob(pattern):
            if path.endswith(".md") or path.endswith(".txt"):
                with open(path, "r", encoding="utf8") as f:
                    text = f.read()
                filename = os.path.basename(path)
                docs.append((filename, text))
        return docs

    # -----------------------------------------------------------
    # Index Construction (Phase 1)
    # -----------------------------------------------------------

    def build_index(self, documents):
        """
        TODO (Phase 1):
        Build a tiny inverted index mapping lowercase words to the documents
        they appear in.

        Example structure:
        {
            "token": ["AUTH.md", "API_REFERENCE.md"],
            "database": ["DATABASE.md"]
        }

        Keep this simple: split on whitespace, lowercase tokens,
        ignore punctuation if needed.
        """

        index = {}
        for filename, text in documents:
            # Track tokens already seen in this document so each filename
            # appears at most once per token.
            seen_tokens = set()

            for token in self._tokenize(text):
                if not token or token in seen_tokens:
                    continue

                seen_tokens.add(token)
                index.setdefault(token, []).append(filename)

        return index

    # -----------------------------------------------------------
    # Scoring and Retrieval (Phase 1)
    # -----------------------------------------------------------

    def score_document(self, query, text):
        """
        TODO (Phase 1):
        Return a simple relevance score for how well the text matches the query.

        Suggested baseline:
        - Convert query into lowercase words
        - Count how many appear in the text
        - Return the count as the score
        """
        # TODO: implement scoring

        query_tokens = set(self._tokenize(query, remove_stopwords=True))
        text_tokens = set(self._tokenize(text))
        score = len(query_tokens & text_tokens)
        return score

    def retrieve(self, query, top_k=3):
        """
        TODO (Phase 1):
        Use the index and scoring function to select top_k relevant document snippets.

        Return a list of (filename, text) sorted by score descending.
        """
        results = []
        # TODO: implement retrieval logic
        query_tokens = set(self._tokenize(query, remove_stopwords=True))

        # If nothing meaningful remains after stopword filtering,
        # retrieval should refuse instead of returning arbitrary snippets.
        if not query_tokens:
            return []

        candidate_filenames = set()
        for token in query_tokens:
            for filename in self.index.get(token, []):
                candidate_filenames.add(filename)  

        for filename, text in self.documents:
            if filename in candidate_filenames:
                score = self.score_document(query, text)
                coverage = score / len(query_tokens)
                if score >= 2 or coverage >= 0.5:
                    results.append((filename, text, score))

        results.sort(key=lambda x: x[2], reverse=True)
        return [(filename, text) for filename, text, _ in results[:top_k]]
    

    # -----------------------------------------------------------
    # Answering Modes
    # -----------------------------------------------------------

    def answer_retrieval_only(self, query, top_k=3):
        """
        Phase 1 retrieval only mode.
        Returns short snippets and filenames with no LLM involved.
        """
        snippets = self.retrieve(query, top_k=top_k)

        if not snippets:
            return NO_INFO_RESPONSE
        ## FIXED: added formatting to make it more readable and concise
        formatted = []
        for filename, text in snippets:
            # Keep output concise by showing only a small preview per file.
            non_empty_lines = [line.strip() for line in text.splitlines() if line.strip()]
            preview = "\n".join(non_empty_lines[:4])
            max_chars = 500
            if len(preview) > max_chars:
                preview = preview[:max_chars].rstrip() + "..."
            formatted.append(f"[{filename}]\n{preview}\n")

        return "\n---\n".join(formatted)

    def answer_rag(self, query, top_k=3):
        """
        Phase 2 RAG mode.
        Uses student retrieval to select snippets, then asks Gemini
        to generate an answer using only those snippets.
        """
        if self.llm_client is None:
            raise RuntimeError(
                "RAG mode requires an LLM client. Provide a GeminiClient instance."
            )

        snippets = self.retrieve(query, top_k=top_k)

        if not snippets:
            return NO_INFO_RESPONSE

        return self.llm_client.answer_from_snippets(query, snippets)

    # -----------------------------------------------------------
    # Bonus Helper: concatenated docs for naive generation mode
    # -----------------------------------------------------------

    def full_corpus_text(self):
        """
        Returns all documents concatenated into a single string.
        This is used in Phase 0 for naive 'generation only' baselines.
        """
        return "\n\n".join(text for _, text in self.documents)
