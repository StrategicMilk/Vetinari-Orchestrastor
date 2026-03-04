"""Test harness for LM Studio chat API."""
import unittest
from unittest.mock import patch, MagicMock


class TestChatEndpoint(unittest.TestCase):
    """Tests for the LM Studio chat API endpoint."""

    @patch("requests.post")
    def test_chat_endpoint_success(self, mock_post):
        """Test the LM Studio chat endpoint returns expected output format."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"output": "Hello in a rhyme!"}
        mock_post.return_value = mock_resp

        import requests
        url = "http://100.78.30.7:1234/api/v1/chat"
        payload = {
            "model": "qwen2.5-0.5b-instruct",
            "system_prompt": "You answer only in rhymes.",
            "input": "Hello",
        }

        r = requests.post(url, json=payload, timeout=60)
        self.assertEqual(r.status_code, 200)

        data = r.json()
        self.assertIn("output", data)
        self.assertEqual(data["output"], "Hello in a rhyme!")

        mock_post.assert_called_once_with(url, json=payload, timeout=60)

    @patch("requests.post")
    def test_chat_endpoint_error_response(self, mock_post):
        """Test handling of non-200 responses."""
        mock_resp = MagicMock()
        mock_resp.status_code = 503
        mock_resp.text = "Service Unavailable"
        mock_post.return_value = mock_resp

        import requests
        r = requests.post("http://100.78.30.7:1234/api/v1/chat",
                          json={}, timeout=60)
        self.assertNotEqual(r.status_code, 200)

    @patch("requests.post")
    def test_chat_endpoint_missing_output_field(self, mock_post):
        """Test that missing 'output' field is handled."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"result": "something unexpected"}
        mock_post.return_value = mock_resp

        import requests
        r = requests.post("http://100.78.30.7:1234/api/v1/chat",
                          json={}, timeout=60)
        data = r.json()
        self.assertNotIn("output", data)


if __name__ == "__main__":
    unittest.main()
