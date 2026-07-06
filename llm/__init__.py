"""LLM integration: the Groq client, a structured-output protocol, and prompts.

The rest of the codebase depends on the :class:`~llm.base.StructuredLLM` protocol
rather than the concrete :class:`~llm.groq_client.GroqClient`, so grading and
narrative generation can be unit-tested with a fake client and no network.
"""
