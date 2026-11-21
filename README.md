# Day 02 Lab: Multimodal RAG over technical manuals

Retrieve by text, answer with the artefact. Provenance (document > page > unit id) is mandatory.

## 1. Vector DB (terminal 1)
```bash
docker compose up -d qdrant   # dashboard: http://localhost:6333/dashboard
```
Or set `VECTOR_BACKEND=memory` in `rag/.env` to skip Docker.

## 2. RAG service (terminal 2)
```bash
cd rag
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # then add OPENROUTER_API_KEY
python scripts/check_models.py
```

## 3. Build the index (terminal 2)
```bash
python scripts/build_index.py
```

## 4. Serve the API (terminal 2)
```bash
uvicorn app.server:app --reload --port 8000
```

## 5. Evaluate (terminal 3)
```bash
cd rag && source .venv/bin/activate
python -m eval.run_eval
```

## 6. UI (terminal 4)
```bash
cd web
cp .env.example .env.local
npm install && npm run dev    # http://localhost:3000
```
