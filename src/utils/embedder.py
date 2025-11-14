from sentence_transformers import SentenceTransformer
from typing import List
from loguru import logger
from dataclasses import dataclass, field
from ..config.setting import settings
import re
import numpy as np

@dataclass
class Embedder:
    model_name: str = field(default=settings.EMBEDDING_MODEL_NAME)
    model: SentenceTransformer = field(init=False)

    def __post_init__(self):
        """Initialize the embedding model after the dataclass is created."""
        try:
            self.model = SentenceTransformer(self.model_name)
            logger.info(f"Embedding model '{self.model_name}' loaded successfully.")
        except Exception as e:
            logger.error(f"Error loading embedding model '{self.model_name}': {e}")
            raise

    @staticmethod
    def _clean_text(text: str) -> str:
        """Clean the input text by removing extra whitespace."""
        return re.sub(r'\s+', ' ', text).strip()

    def embed_texts(self, texts: List[str], clean: bool = True) -> List[np.ndarray]:
        """Generate embeddings for a list of texts, returning a list of NumPy arrays.

        Args:
            texts (List[str]): List of texts to embed.
            clean (bool): Whether to clean the text before embedding. Defaults to True.

        Returns:
            List[np.ndarray]: List of embeddings, one per text.
        """
        embeddings_list = []
        try:
            for text in texts:
                if clean:
                    text = self._clean_text(text)
                embedding = self.model.encode(text, show_progress_bar=True)
                embeddings_list.append(np.array(embedding))
            logger.info(f"Generated embeddings for {len(texts)} texts.")
            return embeddings_list
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise
