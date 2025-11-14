from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, Filter
from loguru import logger
from typing import List, Optional, Dict


class VectorConnection:
    def __init__(self, host: str, port: int, collection_name: str, vector_size: int = 1536):
        """Initialize the Qdrant client and ensure the collection exists.

        Args:
            host (str): Qdrant host address.
            port (int): Qdrant port number.
            collection_name (str): Name of the collection to connect to.
            vector_size (int): Dimension of the vectors. Defaults to 1536.
        """
        try:
            self.client = QdrantClient(host=host, port=port)
            self.collection_name = collection_name
            self.vector_size = vector_size

            # Ensure collection exists
            if not self._collection_exists():
                self.create_collection()
            else:
                logger.info(f"Connected to existing collection '{collection_name}'.")
        except Exception as e:
            logger.error(f"Error initializing Qdrant client: {e}")
            raise

    def _collection_exists(self) -> bool:
        """Check if the collection exists in Qdrant."""
        try:
            self.client.get_collection(self.collection_name)
            return True
        except Exception:
            return False

    def create_collection(self):
        """Create a new collection in Qdrant."""
        try:
            self.client.recreate_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=self.vector_size, distance=Distance.COSINE)
            )
            logger.info(f"Collection '{self.collection_name}' created successfully.")
        except Exception as e:
            logger.error(f"Error creating collection '{self.collection_name}': {e}")
            raise

    def search_vectors(
        self,
        query_vector: List[float],
        top_k: int = 5,
        filter: Optional[Dict] = None
    ) -> List[Dict]:
        """Search for similar vectors in the collection.

        Args:
            query_vector (List[float]): The vector to search for.
            top_k (int): Number of top results to return.
            filter (Optional[Dict]): Optional filter conditions.

        Returns:
            List[Dict]: List of search results with IDs and distances.
        """
        try:
            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=top_k,
                filter=filter
            )
            return [{"id": r.id, "score": r.score, "payload": r.payload} for r in results]
        except Exception as e:
            logger.error(f"Error searching vectors: {e}")
            raise
