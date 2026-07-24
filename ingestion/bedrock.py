"""Bedrock visual model factory for Claude Haiku enrichment."""

from __future__ import annotations

from langchain_aws import ChatBedrockConverse

from ingestion.models import TableValidation, VisualAnalysis


def get_visual_model(settings, max_tokens: int | None = None) -> ChatBedrockConverse:
    from botocore.config import Config

    return ChatBedrockConverse(
        model_id=settings.bedrock_visual_model_id,
        provider="amazon",
        temperature=0,
        max_tokens=max_tokens or settings.bedrock_visual_max_tokens,
        region_name=settings.aws_region,
        config=Config(
            read_timeout=180,
            connect_timeout=15,
            retries={"max_attempts": 2, "mode": "adaptive"},
        ),
    )


def get_visual_analysis_model(settings) -> ChatBedrockConverse:
    return get_visual_model(settings).with_structured_output(VisualAnalysis)


def get_table_validation_model(settings) -> ChatBedrockConverse:
    return get_visual_model(settings).with_structured_output(TableValidation)
