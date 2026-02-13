from __future__ import annotations

from typing import Any, Dict, Optional


class LLMClient:
    """
    Stub LLM client interface for strategy inference.

    Currently returns None to force deterministic fallback.
    Future implementation should integrate with actual LLM API.
    """

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize LLM client.

        Args:
            api_key: Optional API key for LLM service.
                     If None, client will always return None (fallback mode).
        """
        self.api_key = api_key
        self._enabled = api_key is not None

    def infer_strategy(self, prompt_payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Infer ML strategy from dataset profile using LLM.

        Args:
            prompt_payload: Dictionary containing dataset profile information
                           (from profile_to_prompt_dict).

        Returns:
            Optional dictionary with strategy specification, or None if:
            - LLM is not configured (no API key)
            - LLM call fails
            - Response validation fails

        Expected response schema:
        {
            "problem_type": str,
            "target_column": Optional[str],
            "metric": Optional[str],
            "recommended_models": List[str],
            "preprocessing": List[str],
            "reasoning": Dict[str, Any]
        }
        """
        # Stub implementation: always return None to force deterministic fallback
        return None
