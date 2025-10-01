Embedding options

1. OpenAI (recommended for best quality if you have a key)
   - Set OPENAI_API_KEY in environment
   - EMBEDDING_MODEL default: text-embedding-3-small (override via env var)

2. sentence-transformers (offline)
   - Installed through pip: sentence-transformers
   - Uses model 'all-MiniLM-L6-v2' by default for speed and small memory footprint.
   - Good for demo and offline testing.

If you're using the project in production, pick the provider that matches your infra requirements.
