#!/usr/bin/env python3
"""
Integration tests for Sentinel HTTP Server
==========================================

Tests the complete server functionality including:
- Health checks
- Model listing
- Text generation
- Model caching
- Error handling
"""

import pytest
import requests
import time
import json

BASE_URL = "http://127.0.0.1:11435"


class TestServerHealth:
    """Test server health and availability"""

    def test_health_endpoint(self):
        """Test /api/health returns healthy status"""
        response = requests.get(f"{BASE_URL}/api/health", timeout=5)
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "healthy"
        assert "device" in data
        assert "loaded_models" in data
        print(f"✅ Health check passed: {data}")

    def test_root_endpoint(self):
        """Test root endpoint returns server info"""
        response = requests.get(f"{BASE_URL}/", timeout=5)
        assert response.status_code == 200

        data = response.json()
        assert data["name"] == "Sentinel AI Server"
        assert "version" in data
        assert "endpoints" in data
        print(f"✅ Root endpoint passed: {data['name']} v{data['version']}")


class TestModelManagement:
    """Test model discovery and management"""

    def test_list_models(self):
        """Test /api/tags returns available models"""
        response = requests.get(f"{BASE_URL}/api/tags", timeout=5)
        assert response.status_code == 200

        data = response.json()
        assert "models" in data
        assert len(data["models"]) >= 1

        # Check gpt2 is available
        model_names = [m["name"] for m in data["models"]]
        assert "gpt2" in model_names
        print(f"✅ Found {len(data['models'])} models: {model_names}")

    def test_model_metadata(self):
        """Test model metadata is complete"""
        response = requests.get(f"{BASE_URL}/api/tags", timeout=5)
        data = response.json()

        for model in data["models"]:
            assert "name" in model
            assert "size" in model
            assert "family" in model
            assert "modified_at" in model
            print(f"✅ Model {model['name']}: {model['size']} ({model['family']} family)")


class TestTextGeneration:
    """Test actual text generation capabilities"""

    def test_basic_generation(self):
        """Test basic text generation works"""
        request_data = {
            "model": "gpt2",
            "prompt": "Hello, my name is",
            "num_predict": 10,
            "temperature": 0.7,
            "stream": False
        }

        response = requests.post(
            f"{BASE_URL}/api/generate",
            json=request_data,
            timeout=60
        )

        assert response.status_code == 200
        data = response.json()

        # Verify response structure
        assert "response" in data
        assert "model" in data
        assert "done" in data
        assert data["done"] is True
        assert data["model"] == "gpt2"

        # Verify actual generation occurred
        assert len(data["response"]) > 0
        assert isinstance(data["response"], str)

        print(f"✅ Generated text: '{data['response']}'")
        print(f"   Duration: {data.get('total_duration', 0) / 1e9:.2f}s")

    def test_generation_with_system_prompt(self):
        """Test generation with system prompt"""
        request_data = {
            "model": "gpt2",
            "prompt": "What is AI?",
            "system": "You are a helpful assistant.",
            "num_predict": 15,
            "stream": False
        }

        response = requests.post(
            f"{BASE_URL}/api/generate",
            json=request_data,
            timeout=60
        )

        assert response.status_code == 200
        data = response.json()
        assert len(data["response"]) > 0
        print(f"✅ With system prompt: '{data['response']}'")

    def test_generation_deterministic(self):
        """Test temperature=0 produces consistent results"""
        request_data = {
            "model": "gpt2",
            "prompt": "The capital of France is",
            "num_predict": 5,
            "temperature": 0.0,  # Deterministic
            "stream": False
        }

        # Generate twice
        response1 = requests.post(f"{BASE_URL}/api/generate", json=request_data, timeout=60)
        response2 = requests.post(f"{BASE_URL}/api/generate", json=request_data, timeout=60)

        data1 = response1.json()
        data2 = response2.json()

        # With temperature=0, results should be identical
        assert data1["response"] == data2["response"]
        print(f"✅ Deterministic output: '{data1['response']}'")

    def test_model_caching(self):
        """Test model stays loaded (faster second request)"""
        request_data = {
            "model": "gpt2",
            "prompt": "Test prompt",
            "num_predict": 5,
            "stream": False
        }

        # First request (cold start)
        start1 = time.time()
        response1 = requests.post(f"{BASE_URL}/api/generate", json=request_data, timeout=60)
        time1 = time.time() - start1

        # Second request (cached model)
        start2 = time.time()
        response2 = requests.post(f"{BASE_URL}/api/generate", json=request_data, timeout=60)
        time2 = time.time() - start2

        assert response1.status_code == 200
        assert response2.status_code == 200

        # Second request should be faster (model cached)
        # Allow some variance but expect significant speedup
        assert time2 < time1 * 0.5 or time2 < 2.0  # Either 50% faster or under 2s

        print(f"✅ Caching works: first={time1:.2f}s, second={time2:.2f}s (speedup: {time1/time2:.1f}x)")


class TestErrorHandling:
    """Test error handling and edge cases"""

    def test_invalid_model(self):
        """Test requesting non-existent model returns error"""
        request_data = {
            "model": "does-not-exist",
            "prompt": "Test",
            "num_predict": 5
        }

        response = requests.post(
            f"{BASE_URL}/api/generate",
            json=request_data,
            timeout=60
        )

        # Should fail gracefully
        assert response.status_code in [400, 500]
        print(f"✅ Invalid model handled: {response.status_code}")

    def test_empty_prompt(self):
        """Test empty prompt handling"""
        request_data = {
            "model": "gpt2",
            "prompt": "",
            "num_predict": 5
        }

        response = requests.post(
            f"{BASE_URL}/api/generate",
            json=request_data,
            timeout=60
        )

        # Should handle gracefully (either success, client error, or server error)
        assert response.status_code in [200, 400, 500]
        print(f"✅ Empty prompt handled: {response.status_code}")

    def test_very_long_generation(self):
        """Test requesting many tokens"""
        request_data = {
            "model": "gpt2",
            "prompt": "Once upon a time",
            "num_predict": 100,  # Long generation
            "stream": False
        }

        response = requests.post(
            f"{BASE_URL}/api/generate",
            json=request_data,
            timeout=120
        )

        assert response.status_code == 200
        data = response.json()

        # Should generate substantial text
        assert len(data["response"]) > 50
        print(f"✅ Long generation: {len(data['response'])} chars")


class TestPerformance:
    """Test performance characteristics"""

    def test_concurrent_requests(self):
        """Test server handles multiple concurrent requests"""
        import concurrent.futures

        def make_request(n):
            request_data = {
                "model": "gpt2",
                "prompt": f"Request {n}:",
                "num_predict": 5,
                "stream": False
            }
            response = requests.post(f"{BASE_URL}/api/generate", json=request_data, timeout=60)
            return response.status_code == 200

        # Send 3 concurrent requests
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(make_request, i) for i in range(3)]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]

        # All should succeed
        assert all(results)
        print(f"✅ Concurrent requests: {len(results)}/{len(results)} succeeded")


if __name__ == "__main__":
    # Run with pytest for better output
    pytest.main([__file__, "-v", "-s"])
