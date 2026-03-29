# bookworm
Local deep research

## Modal Setup

### 1. Generate an API key

```bash
echo "BOOKWORM_API_KEY=$(python -c 'import secrets; print(secrets.token_urlsafe(32))')" >> .env
```

### 2. Install Modal and authenticate

```bash
uv sync --group remote
uv run modal setup
```

### 3. Add your API key to Modal

```bash
uv run --env-file=.env modal secret create bookworm-auth BOOKWORM_API_KEY=$BOOKWORM_API_KEY
```

### 4. Deploy

```bash
uv run modal deploy -m bookworm.server.remote
```
