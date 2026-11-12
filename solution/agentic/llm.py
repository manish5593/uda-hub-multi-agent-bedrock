"""Factory helpers for chat and embedding models.

Every agent gets its models from here, so switching provider is a configuration change:
set UDAHUB_LLM_PROVIDER to "openai" or "bedrock", and optionally UDAHUB_EMBEDDING_PROVIDER to
"openai", "bedrock" or "sagemaker" (see agentic/config.py).
"""
from functools import lru_cache

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel

from agentic import config


def _bedrock_config():
    """Retries for Bedrock calls. A new AWS account's request limits for Anthropic models are low, and the default
    4 attempts gave up within seconds (ThrottlingException), so a burst of messages got the fallback reply.
    Adaptive mode backs off and also slows the client down when it is throttled."""
    from botocore.config import Config
    return Config(retries={"max_attempts": config.BEDROCK_MAX_ATTEMPTS, "mode": "adaptive"},
                  read_timeout=120, connect_timeout=10)


@lru_cache(maxsize=1)
def _bedrock_rate_limiter():
    """One limiter for every Bedrock chat model in this process, so all agents share the account's quota."""
    if config.BEDROCK_REQUESTS_PER_MINUTE <= 0:
        return None
    from langchain_core.rate_limiters import InMemoryRateLimiter
    return InMemoryRateLimiter(requests_per_second=config.BEDROCK_REQUESTS_PER_MINUTE / 60,
                               check_every_n_seconds=0.1, max_bucket_size=2)


@lru_cache(maxsize=None)
def get_llm(temperature: float = 0.0) -> BaseChatModel:
    if config.LLM_PROVIDER == "bedrock":
        from langchain_aws import ChatBedrockConverse
        # The Converse API supports the tool calling and structured output the agents rely on
        return ChatBedrockConverse(model=config.CHAT_MODEL, region_name=config.AWS_REGION,
                                   temperature=temperature, max_tokens=config.BEDROCK_MAX_TOKENS, config=_bedrock_config(),
                                   rate_limiter=_bedrock_rate_limiter())
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(model=config.CHAT_MODEL, temperature=temperature)


@lru_cache(maxsize=None)
def get_embeddings() -> Embeddings:
    if config.EMBEDDING_PROVIDER == "sagemaker":
        from agentic.sagemaker_embeddings import SageMakerEndpointEmbeddings
        return SageMakerEndpointEmbeddings(config.SAGEMAKER_ENDPOINT, region_name=config.AWS_REGION)
    if config.EMBEDDING_PROVIDER == "bedrock":
        from langchain_aws import BedrockEmbeddings
        return BedrockEmbeddings(model_id=config.EMBEDDING_MODEL, region_name=config.AWS_REGION, normalize=True,
                                 config=_bedrock_config())
    from langchain_openai import OpenAIEmbeddings
    return OpenAIEmbeddings(model=config.EMBEDDING_MODEL)
