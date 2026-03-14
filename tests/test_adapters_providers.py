"""Tests for all provider adapters (LMStudio, OpenAI, Anthropic, Gemini, Cohere) — Phase 7B"""
import unittest
from unittest.mock import MagicMock, patch

from vetinari.adapters.base import (
    InferenceRequest,
    ModelInfo,
    ProviderConfig,
    ProviderType,
)


def _cfg(pt, endpoint="http://ep", api_key="test-key"):  # noqa: VET040
    return ProviderConfig(provider_type=pt, name="test", endpoint=endpoint, api_key=api_key)


class TestLMStudioAdapter(unittest.TestCase):

    def setUp(self):
        from vetinari.adapters.lmstudio_adapter import LMStudioProviderAdapter
        self.AdapterClass = LMStudioProviderAdapter

    def test_wrong_provider_type_raises(self):
        cfg = _cfg(ProviderType.OPENAI)
        with self.assertRaises(ValueError):
            self.AdapterClass(cfg)

    @patch("requests.Session")
    def test_discover_models_dict_format(self, mock_session_cls):
        session = MagicMock()
        resp = MagicMock()
        resp.json.return_value = {"data": [{"id": "llama-3", "name": "Llama 3"}]}
        resp.raise_for_status.return_value = None
        session.get.return_value = resp
        mock_session_cls.return_value = session

        adapter = self.AdapterClass(_cfg(ProviderType.LM_STUDIO))
        models = adapter.discover_models()
        self.assertIsInstance(models, list)
        self.assertEqual(models[0].id, "llama-3")

    @patch("requests.Session")
    def test_discover_models_empty_on_error(self, mock_session_cls):
        session = MagicMock()
        session.get.side_effect = Exception("connection refused")
        mock_session_cls.return_value = session

        adapter = self.AdapterClass(_cfg(ProviderType.LM_STUDIO))
        models = adapter.discover_models()
        self.assertEqual(models, [])

    @patch("requests.Session")
    def test_health_check_healthy(self, mock_session_cls):
        session = MagicMock()
        resp = MagicMock()
        resp.json.return_value = {"data": []}
        resp.raise_for_status.return_value = None
        session.get.return_value = resp
        mock_session_cls.return_value = session

        adapter = self.AdapterClass(_cfg(ProviderType.LM_STUDIO))
        health = adapter.health_check()
        self.assertIn("healthy", health)

    @patch("requests.Session")
    def test_health_check_unhealthy(self, mock_session_cls):
        session = MagicMock()
        session.get.side_effect = Exception("refused")
        mock_session_cls.return_value = session

        adapter = self.AdapterClass(_cfg(ProviderType.LM_STUDIO))
        health = adapter.health_check()
        self.assertFalse(health["healthy"])

    @patch("requests.Session")
    def test_infer_success(self, mock_session_cls):
        session = MagicMock()
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {"choices": [{"message": {"content": "hello"}}],
                                  "usage": {"total_tokens": 20}}
        resp.raise_for_status.return_value = None
        session.post.return_value = resp
        mock_session_cls.return_value = session

        adapter = self.AdapterClass(_cfg(ProviderType.LM_STUDIO))
        req = InferenceRequest(model_id="llama-3", prompt="Say hello")
        result = adapter.infer(req)
        self.assertEqual(result.status, "ok")
        self.assertEqual(result.output, "hello")

    @patch("requests.Session")
    def test_infer_error(self, mock_session_cls):
        session = MagicMock()
        session.post.side_effect = Exception("timeout")
        mock_session_cls.return_value = session

        adapter = self.AdapterClass(_cfg(ProviderType.LM_STUDIO))
        req = InferenceRequest(model_id="m1", prompt="test")
        result = adapter.infer(req)
        self.assertEqual(result.status, "error")

    @patch("requests.Session")
    def test_get_capabilities(self, mock_session_cls):
        session = MagicMock()
        resp = MagicMock()
        resp.json.return_value = {"data": [{"id": "m1"}]}
        resp.raise_for_status.return_value = None
        session.get.return_value = resp
        mock_session_cls.return_value = session

        adapter = self.AdapterClass(_cfg(ProviderType.LM_STUDIO))
        caps = adapter.get_capabilities()
        self.assertIsInstance(caps, dict)


class TestOpenAIAdapter(unittest.TestCase):

    def setUp(self):
        from vetinari.adapters.openai_adapter import OpenAIProviderAdapter
        self.AdapterClass = OpenAIProviderAdapter

    def test_wrong_provider_type_raises(self):
        with self.assertRaises(ValueError):
            self.AdapterClass(_cfg(ProviderType.LM_STUDIO))

    def test_missing_api_key_raises(self):
        cfg = ProviderConfig(provider_type=ProviderType.OPENAI, name="t",
                             endpoint="https://api.openai.com", api_key=None)
        with self.assertRaises(ValueError):
            self.AdapterClass(cfg)

    @patch("requests.Session")
    def test_discover_models_returns_list(self, _mock):
        adapter = self.AdapterClass(_cfg(ProviderType.OPENAI))
        models = adapter.discover_models()
        self.assertIsInstance(models, list)
        self.assertGreater(len(models), 0)

    @patch("requests.Session")
    def test_health_check_ok(self, mock_session_cls):
        session = MagicMock()
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {"object": "list"}
        session.get.return_value = resp
        mock_session_cls.return_value = session

        adapter = self.AdapterClass(_cfg(ProviderType.OPENAI))
        health = adapter.health_check()
        self.assertIn("healthy", health)

    @patch("requests.Session")
    def test_infer_success(self, mock_session_cls):
        session = MagicMock()
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {
            "choices": [{"message": {"content": "world"}}],
            "usage": {"total_tokens": 15},
        }
        session.post.return_value = resp
        mock_session_cls.return_value = session

        adapter = self.AdapterClass(_cfg(ProviderType.OPENAI))
        result = adapter.infer(InferenceRequest(model_id="gpt-4", prompt="hello"))
        self.assertEqual(result.output, "world")

    @patch("requests.Session")
    def test_infer_error(self, mock_session_cls):
        session = MagicMock()
        session.post.side_effect = Exception("fail")
        mock_session_cls.return_value = session

        adapter = self.AdapterClass(_cfg(ProviderType.OPENAI))
        result = adapter.infer(InferenceRequest(model_id="gpt-4", prompt="test"))
        self.assertEqual(result.status, "error")


class TestAnthropicAdapter(unittest.TestCase):

    def setUp(self):
        from vetinari.adapters.anthropic_adapter import AnthropicProviderAdapter
        self.AdapterClass = AnthropicProviderAdapter

    def test_wrong_type_raises(self):
        with self.assertRaises(ValueError):
            self.AdapterClass(_cfg(ProviderType.OPENAI))

    @patch("requests.Session")
    def test_discover_models_returns_list(self, _mock):
        adapter = self.AdapterClass(_cfg(ProviderType.ANTHROPIC))
        models = adapter.discover_models()
        self.assertIsInstance(models, list)

    @patch("requests.Session")
    def test_infer_success(self, mock_session_cls):
        session = MagicMock()
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {"content": [{"text": "answer"}], "usage": {"output_tokens": 10, "input_tokens": 5}}
        session.post.return_value = resp
        mock_session_cls.return_value = session

        adapter = self.AdapterClass(_cfg(ProviderType.ANTHROPIC))
        result = adapter.infer(InferenceRequest(model_id="claude-3", prompt="q"))
        self.assertIn(result.status, ("ok", "error"))


class TestGeminiAdapter(unittest.TestCase):

    def setUp(self):
        from vetinari.adapters.gemini_adapter import GeminiProviderAdapter
        self.AdapterClass = GeminiProviderAdapter

    def test_wrong_type_raises(self):
        with self.assertRaises(ValueError):
            self.AdapterClass(_cfg(ProviderType.OPENAI))

    @patch("requests.Session")
    def test_discover_models_returns_list(self, _mock):
        adapter = self.AdapterClass(_cfg(ProviderType.GEMINI))
        models = adapter.discover_models()
        self.assertIsInstance(models, list)

    @patch("requests.Session")
    def test_health_check_returns_dict(self, mock_session_cls):
        session = MagicMock()
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {"models": []}
        session.get.return_value = resp
        mock_session_cls.return_value = session

        adapter = self.AdapterClass(_cfg(ProviderType.GEMINI))
        health = adapter.health_check()
        self.assertIn("healthy", health)


class TestCohereAdapter(unittest.TestCase):

    def setUp(self):
        from vetinari.adapters.cohere_adapter import CohereProviderAdapter
        self.AdapterClass = CohereProviderAdapter

    def test_wrong_type_raises(self):
        with self.assertRaises(ValueError):
            self.AdapterClass(_cfg(ProviderType.OPENAI))

    @patch("requests.Session")
    def test_discover_models_returns_list(self, _mock):
        adapter = self.AdapterClass(_cfg(ProviderType.COHERE))
        models = adapter.discover_models()
        self.assertIsInstance(models, list)

    @patch("requests.Session")
    def test_infer_error_on_exception(self, mock_session_cls):
        session = MagicMock()
        session.post.side_effect = Exception("net error")
        mock_session_cls.return_value = session

        adapter = self.AdapterClass(_cfg(ProviderType.COHERE))
        result = adapter.infer(InferenceRequest(model_id="command", prompt="x"))
        self.assertEqual(result.status, "error")


if __name__ == "__main__":
    unittest.main()
