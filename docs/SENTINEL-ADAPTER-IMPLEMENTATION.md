# Sentinel Adapter Implementation Guide

## Quick Summary

This document provides ready-to-use code for integrating Sentinel-AI's pre-trained models into Continuum's AI Provider system.

**Copy-paste the files below, run `npm start`, and PersonaUsers will use pre-trained models!**

---

## Step 1: Create Sentinel Adapter

**File**: `src/debug/jtag/daemons/ai-provider-daemon/adapters/sentinel/shared/SentinelAdapter.ts`

```typescript
/**
 * Sentinel Adapter - Pre-trained Model Integration
 * =================================================
 *
 * Adapter for Sentinel-AI pre-trained models (TinyLlama, Phi-2, CodeLlama, etc.)
 * Provides local inference with models from the Sentinel-AI model zoo.
 *
 * Features:
 * - Text generation with pre-trained transformers
 * - No external API dependencies
 * - Privacy-first (models run locally)
 * - Adaptive architecture support (future)
 *
 * Python Bridge: /Volumes/FlashGordon/cambrian/sentinel-ai/scripts/continuum_inference.py
 */

import type {
  TextGenerationRequest,
  TextGenerationResponse,
  UsageMetrics,
} from '../../../shared/AIProviderTypes';
import {
  estimateTokenCount,
  createRequestId,
  AIProviderError,
} from '../../../shared/AIProviderTypes';
import { BaseAIProviderAdapter } from '../../../shared/BaseAIProviderAdapter';
import { exec } from 'child_process';
import { promisify } from 'util';

const execAsync = promisify(exec);

/**
 * Sentinel Adapter - Executes Python scripts to load and run pre-trained models
 */
export class SentinelAdapter extends BaseAIProviderAdapter {
  private readonly sentinelPath = '/Volumes/FlashGordon/cambrian/sentinel-ai';
  private readonly pythonWrapper = 'experiments/run_with_continuum_python.sh';

  // For now, use stub for fast testing
  // Once proven, swap to real inference: 'scripts/continuum_inference.py'
  private readonly inferenceScript = '/tmp/test_sentinel_stub.py';

  constructor() {
    super('sentinel');
    console.log('🧬 Sentinel Adapter initialized');
    console.log(`   Path: ${this.sentinelPath}`);
    console.log(`   Script: ${this.inferenceScript}`);
  }

  async generateText(request: TextGenerationRequest): Promise<TextGenerationResponse> {
    const requestId = createRequestId();
    const startTime = Date.now();

    console.log(`🧬 Sentinel: Generating text (model: ${request.model || 'tinyllama-chat'})`);

    try {
      // Build arguments for Python script
      const messagesJson = JSON.stringify(request.messages).replace(/"/g, '\\"');
      const model = request.model || 'tinyllama-chat';
      const temperature = request.temperature ?? 0.7;
      const maxTokens = request.maxTokens ?? 150;

      // Execute Python script
      const command = `cd "${this.sentinelPath}" && python3 "${this.inferenceScript}" --model "${model}" --messages "${messagesJson}" --temperature ${temperature} --max-tokens ${maxTokens}`;

      console.log(`🔧 Executing: ${command.substring(0, 150)}...`);

      const { stdout, stderr } = await execAsync(command, {
        timeout: 120000, // 2 minutes
        maxBuffer: 10 * 1024 * 1024 // 10MB
      });

      if (stderr && !stderr.includes('FutureWarning')) {
        console.warn(`⚠️  Sentinel stderr: ${stderr.substring(0, 200)}`);
      }

      // Parse JSON response from Python
      const response = JSON.parse(stdout.trim());

      if (response.error) {
        throw new Error(`Python error: ${response.error}`);
      }

      // Calculate usage metrics
      const promptTokens = request.messages.reduce(
        (sum, msg) => sum + estimateTokenCount(msg.content),
        0
      );
      const completionTokens = estimateTokenCount(response.text);

      const result: TextGenerationResponse = {
        text: response.text,
        model: response.metadata?.model || model,
        finishReason: 'stop',
        usage: {
          promptTokens,
          completionTokens,
          totalTokens: promptTokens + completionTokens
        }
      };

      const duration = Date.now() - startTime;
      console.log(`✅ Sentinel: Generated ${completionTokens} tokens in ${duration}ms`);

      // Log if using stub (for debugging)
      if (response.metadata?.stub) {
        console.log(`ℹ️  Sentinel: Using stub response (swap to real inference when ready)`);
      }

      return result;

    } catch (error) {
      const duration = Date.now() - startTime;
      console.error(`❌ Sentinel: Generation failed after ${duration}ms:`, error);

      throw new AIProviderError(
        `Sentinel generation failed: ${error instanceof Error ? error.message : 'Unknown error'}`,
        'GENERATION_ERROR',
        {
          provider: 'sentinel',
          model: request.model || 'tinyllama-chat',
          requestId,
          duration
        }
      );
    }
  }
}
```

---

## Step 2: Register Sentinel Adapter

**File**: `src/debug/jtag/daemons/ai-provider-daemon/server/AIProviderDaemonServer.ts`

Find the adapter initialization section and add:

```typescript
// BEFORE (existing code):
private initializeAdapters(): void {
  this.adapters.set('anthropic', new AnthropicAdapter());
  this.adapters.set('openai', new OpenAIAdapter());
  this.adapters.set('ollama', new OllamaAdapter());
  // ... other adapters
}

// AFTER (add Sentinel):
import { SentinelAdapter } from '../adapters/sentinel/shared/SentinelAdapter';

private initializeAdapters(): void {
  this.adapters.set('anthropic', new AnthropicAdapter());
  this.adapters.set('openai', new OpenAIAdapter());
  this.adapters.set('ollama', new OllamaAdapter());
  this.adapters.set('sentinel', new SentinelAdapter());  // ADD THIS LINE
  // ... other adapters
}
```

---

## Step 3: Add Sentinel to Provider Types

**File**: `src/debug/jtag/daemons/ai-provider-daemon/shared/AIProviderTypes.ts`

Find the AIProvider type and add 'sentinel':

```typescript
// BEFORE:
export type AIProvider =
  | 'anthropic'
  | 'openai'
  | 'ollama'
  | 'groq'
  | 'fireworks'
  | 'together'
  | 'xai'
  | 'deepseek';

// AFTER:
export type AIProvider =
  | 'anthropic'
  | 'openai'
  | 'ollama'
  | 'sentinel'  // ADD THIS LINE
  | 'groq'
  | 'fireworks'
  | 'together'
  | 'xai'
  | 'deepseek';
```

---

## Step 4: Configure PersonaUser to Use Sentinel

**Option A**: Modify existing PersonaUser

In the database or user creation, set:

```typescript
{
  modelConfig: {
    provider: 'sentinel',
    model: 'tinyllama-chat',  // or 'phi-2', 'codellama-7b', 'distilgpt2'
    temperature: 0.7,
    maxTokens: 150
  }
}
```

**Option B**: Create new PersonaUser with Sentinel

```bash
cd /Users/joel/Development/continuum/src/debug/jtag
./jtag user/create \
  --displayName="TinyLlama AI" \
  --userType="persona" \
  --modelConfig='{"provider":"sentinel","model":"tinyllama-chat","temperature":0.7,"maxTokens":150}'
```

---

## Step 5: Test Integration

### 5.1 Deploy

```bash
cd /Users/joel/Development/continuum/src/debug/jtag
npm start  # Wait 90+ seconds
```

### 5.2 Verify Sentinel Provider Registered

Check logs for:
```
🧬 Sentinel Adapter initialized
   Path: /Volumes/FlashGordon/cambrian/sentinel-ai
   Script: /tmp/test_sentinel_stub.py
```

### 5.3 Send Test Message

```bash
# Get room ID
./jtag data/list --collection=rooms --filter='{"uniqueId":"general"}' --limit=1

# Send message (replace ROOM_ID)
./jtag debug/chat-send --roomId="<ROOM_ID>" --message="Hello, TinyLlama AI! Can you introduce yourself?"
```

### 5.4 Verify Response

Check logs for:
```
🧬 Sentinel: Generating text (model: tinyllama-chat)
🔧 Executing: cd "/Volumes/FlashGordon/cambrian/sentinel-ai" && python3...
✅ Sentinel: Generated X tokens in Yms
ℹ️  Sentinel: Using stub response (swap to real inference when ready)
```

### 5.5 Check Chat Widget

```bash
./jtag screenshot --querySelector="chat-widget" --filename="/tmp/sentinel-test.png"
open /tmp/sentinel-test.png
```

You should see the PersonaUser's response from the stub!

---

## Step 6: Swap to Real Inference (When Ready)

Once the stub is proven working, enable real model inference:

**Edit**: `SentinelAdapter.ts` line 39:

```typescript
// FROM:
private readonly inferenceScript = '/tmp/test_sentinel_stub.py';

// TO:
private readonly inferenceScript = 'scripts/continuum_inference.py';
```

**Redeploy**:
```bash
npm start
```

Now PersonaUsers will use actual TinyLlama models! 🧬🚀

---

## Troubleshooting

### Issue: "Sentinel generation failed"

**Check**:
1. Is stub script present? `ls -lh /tmp/test_sentinel_stub.py`
2. Does it have execute permissions? `chmod +x /tmp/test_sentinel_stub.py`
3. Can you run it manually? `python3 /tmp/test_sentinel_stub.py --model tinyllama-chat --prompt "test"`

### Issue: "Python error" or timeout

**Check**:
1. Is Python environment correct? `which python3`
2. Are dependencies installed? `cd /Volumes/FlashGordon/cambrian/sentinel-ai && pip list | grep torch`
3. Does model load? `cd /Volumes/FlashGordon/cambrian/sentinel-ai && experiments/run_with_continuum_python.sh /tmp/test_tinyllama_loading.py`

### Issue: PersonaUser not responding

**Check**:
1. Is PersonaUser configured with sentinel provider? `./jtag data/read --collection=users --id=<USER_ID>`
2. Is AIProviderDaemon running? Check logs for "🧬 Sentinel Adapter initialized"
3. Are there errors in server logs? `tail -f .continuum/sessions/user/shared/*/logs/server.log`

---

## Performance Notes

**Stub (current)**:
- Response time: ~100ms
- Memory: Negligible
- Purpose: Prove integration works

**Real Inference (after swap)**:
- First request: ~5-10s (model loading)
- Subsequent: ~2-3s (generation only)
- Memory: 1-4GB depending on model
- Optimization: Add model caching (keep loaded between requests)

---

## Next Steps

1. ✅ **Prove stub works** - Verify PersonaUser generates responses via Sentinel adapter
2. 🚧 **Fix real inference** - Resolve `model.generate()` hanging issue
3. 🚧 **Add model caching** - Keep models loaded for ~200-500ms response time
4. 🚧 **Test TinyLlama** - Swap to real inference and verify coherent output
5. 📋 **Add more models** - Test Phi-2, CodeLlama
6. 📋 **Adaptive integration** - Enable budget-oriented compression

---

## Files Summary

**New Files** (create these):
1. `src/debug/jtag/daemons/ai-provider-daemon/adapters/sentinel/shared/SentinelAdapter.ts`

**Modified Files** (edit these):
2. `src/debug/jtag/daemons/ai-provider-daemon/server/AIProviderDaemonServer.ts` (add import + registration)
3. `src/debug/jtag/daemons/ai-provider-daemon/shared/AIProviderTypes.ts` (add 'sentinel' to union type)

**Test Files** (already exist):
- `/tmp/test_sentinel_stub.py` - Stub for testing
- `/Volumes/FlashGordon/cambrian/sentinel-ai/scripts/continuum_inference.py` - Real inference (swap later)

---

## Success Criteria

✅ **Phase 1: Stub Integration** (30 minutes)
- [ ] SentinelAdapter compiles without errors
- [ ] AIProviderDaemon registers Sentinel adapter
- [ ] PersonaUser configured with sentinel provider
- [ ] Test message → Stub response appears in chat
- [ ] No errors in server logs

✅ **Phase 2: Real Inference** (1-2 hours)
- [ ] Fix `model.generate()` hanging issue
- [ ] Swap to real inference script
- [ ] TinyLlama generates coherent text
- [ ] Response quality acceptable
- [ ] Performance < 10s per request

✅ **Phase 3: Production Ready** (1 week)
- [ ] Model caching (< 1s response time)
- [ ] Multiple models tested (Phi-2, CodeLlama)
- [ ] Error handling robust
- [ ] Documentation complete
- [ ] Integration tests passing

---

## Architecture Diagram

```
Chat Message
    ↓
PersonaUser.respondToMessage()
    ↓
AIProviderDaemon.generateText({ provider: 'sentinel', model: 'tinyllama-chat' })
    ↓
SentinelAdapter.generateText()
    ↓
exec('python3 /tmp/test_sentinel_stub.py --model tinyllama-chat ...')
    ↓
Python returns JSON: {"text": "Hello! I'm powered by Sentinel-AI...", "metadata": {...}}
    ↓
Parse JSON → TextGenerationResponse
    ↓
PersonaUser.postMessage()
    ↓
Response appears in chat! 🎉
```

---

## The Big Picture

This integration unlocks:

1. **Privacy**: Models run locally, data never leaves machine
2. **Cost**: No API fees
3. **Customization**: Fine-tune models on specific domains
4. **Self-Optimization**: PersonaUsers can compress/grow their own "brains"
5. **Autonomy**: True AI citizens with their own neural architectures

**We're building the foundation for evolvable AI genomes!** 🧬🚀

Once this works, the path to adaptive architectures, LoRA adapter paging, and continuous learning is clear. This is the bridge between Continuum's coordination layer and Sentinel-AI's adaptive transformers.

**Now go make it work!** 💪
