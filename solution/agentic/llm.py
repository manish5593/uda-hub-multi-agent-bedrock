"""Factory helpers for chat and embedding models.

Every agent gets its models from here, so switching provider is a configuration change:
set UDAHUB_LLM_PROVIDER to "openai" or "bedrock", and optionally UDAHUB_EMBEDDING_PROVIDER to
"openai", "bedrock" or "sagemaker" (see agentic/config.py).
"""
from functools import lru_cache

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel

from agentic import config


@lru_cache(maxsize=None)
def get_llm(temperature: float = 0.0) -> BaseChatModel:
    if config.LLM_PROVIDER == "bedrock":
        from langchain_aws import ChatBedrockConverse
        # The Converse API supports the tool calling and structured output the agents rely on
        return ChatBedrockConverse(model=config.CHAT_MODEL, region_name=config.AWS_REGION,
                                   temperature=temperature, max_tokens=2048)
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(model=config.CHAT_MODEL, temperature=temperature)


@lru_cache(maxsize=None)
def get_embeddings() -> Embeddings:
    if config.EMBEDDING_PROVIDER == "sagemaker":
        from agentic.sagemaker_embeddings import SageMakerEndpointEmbeddings
        return SageMakerEndpointEmbeddings(config.SAGEMAKER_ENDPOINT, region_name=config.AWS_REGION)
    if config.EMBEDDING_PROVIDER == "bedrock":
        from langchain_aws import BedrockEmbeddings
        return BedrockEmbeddings(model_id=config.EMBEDDING_MODEL, region_name=config.AWS_REGION, normalize=True)
    from langchain_openai import OpenAIEmbeddings
    return OpenAIEmbeddings(model=config.EMBEDDING_MODEL)
