# Sentinel HTTP Server

Simple HTTP server for Sentinel model inference, following Ollama's API design.

## Quick Start

```bash
# Start server (default port 11435)
./server/start_server.sh

# Or specify custom port
./server/start_server.sh 11436
SENTINEL_PORT=11436 ./server/start_server.sh
```

## API Endpoints

### Generate Text
```bash
POST /api/generate
Content-Type: application/json

{
  "model": "gpt2",
  "prompt": "Once upon a time",
  "system": "You are a helpful assistant",  // optional
  "temperature": 0.7,
  "num_predict": 150,
  "stream": false
}
```

### List Models
```bash
GET /api/tags

Response:
{
  "models": [
    {"name": "gpt2", "size": "124M", ...},
    {"name": "distilgpt2", "size": "82M", ...}
  ]
}
```

### Health Check
```bash
GET /api/health

Response:
{
  "status": "healthy",
  "device": "cuda",
  "loaded_models": ["gpt2"]
}
```

## Configuration

Environment variables:
- `SENTINEL_PORT`: Server port (default: 11435)
- `SENTINEL_HOST`: Bind address (default: 127.0.0.1)

## Integration with Continuum

The Continuum SentinelAdapter automatically starts the server if needed.

**Auto-Start (Default)**:
- First request triggers server auto-start
- Waits up to 30 seconds for server to be ready
- Server keeps running for subsequent requests

**Manual Start (Optional)**:
```bash
./server/start_server.sh
```

**Configuration**:
- `SENTINEL_PORT`: Server port (default: 11435)
- `SENTINEL_PATH`: Path to sentinel-ai project (default: /Volumes/FlashGordon/cambrian/sentinel-ai)

## Available Models

- `gpt2` (124M) - Fast, good for testing
- `distilgpt2` (82M) - Faster, smaller
- `microsoft/phi-2` (2.7B) - Better quality, slower

Models are loaded on-demand and cached in memory.

## Architecture

- Flask HTTP server
- Models stay loaded (no reload per request)
- Non-blocking inference
- Compatible with Ollama adapter pattern
- Multiple Sentinel personas can use different models

## Testing

Run integration tests to verify server functionality:

```bash
# Start server first
./server/start_server.sh

# In another terminal, run tests
./tests/run_tests.sh
```

**Test Coverage**:
- Health checks and server info
- Model discovery and metadata
- Text generation (basic, system prompts, deterministic)
- Model caching (50x speedup verified)
- Error handling (invalid models, empty prompts, long generations)
- Concurrent requests

All 12 tests passing ✅

## Benefits vs Old Design

**Old (shell exec):**
- ❌ Spawn Python process per request
- ❌ Write temp JSON files to disk
- ❌ Parse stdout
- ❌ No connection pooling
- ❌ Blocks during load

**New (HTTP server):**
- ✅ Models stay loaded
- ✅ HTTP requests (fast)
- ✅ Non-blocking
- ✅ Multiple personas supported
- ✅ Health checks
- ✅ Graceful degradation
- ✅ **Fully tested** (12 integration tests)
