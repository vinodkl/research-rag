# Security policy

Please do not open a public issue for a suspected vulnerability or leaked
credential. Use the repository's private GitHub security-advisory reporting
channel when available and include the affected version, impact, minimal
reproduction, and suggested mitigation. Maintainers should acknowledge reports
privately before coordinating disclosure.

The repository is an educational RAG implementation. Its deterministic
guardrails reduce common mistakes but are not a security boundary. A public
deployment is responsible for authentication, authorization, TLS, rate limiting,
moderation, DLP, dependency patching, audit logging, and protection of its corpus
and OpenAI key.
