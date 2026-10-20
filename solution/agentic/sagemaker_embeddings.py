"""Embeddings served from an Amazon SageMaker endpoint.

The endpoint runs Hugging Face Text Embeddings Inference (TEI), deployed with deploy/sagemaker_embeddings.py.
TEI takes {"inputs": [...]} and returns one vector per input.
"""
import json
from typing import List

from langchain_core.embeddings import Embeddings


class SageMakerEndpointEmbeddings(Embeddings):
    def __init__(self, endpoint_name: str, region_name: str, batch_size: int = 32, client=None):
        self.endpoint_name = endpoint_name
        # TEI rejects requests with more inputs than its max client batch size (32 by default)
        self.batch_size = batch_size
        if client is None:
            import boto3
            client = boto3.client("sagemaker-runtime", region_name=region_name)
        self._client = client

    def _embed(self, texts: List[str]) -> List[List[float]]:
        vectors = []
        for start in range(0, len(texts), self.batch_size):
            body = {"inputs": texts[start:start + self.batch_size], "normalize": True, "truncate": True}
            response = self._client.invoke_endpoint(
                EndpointName=self.endpoint_name, ContentType="application/json",
                Accept="application/json", Body=json.dumps(body))
            vectors.extend(json.loads(response["Body"].read()))
        return vectors

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._embed(list(texts))

    def embed_query(self, text: str) -> List[float]:
        return self._embed([text])[0]
