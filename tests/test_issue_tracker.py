"""Tests for the issue tracker integration layer.

Covers the abstract data structures, the factory function, and all three
concrete adapters (GitHub, Linear, Jira) using mocked HTTP responses so
that no real network calls are made.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import requests

from vetinari.integrations import (
    CreateIssueRequest,
    Issue,
    IssuePriority,
    IssueStatus,
    IssueTrackerError,
    create_issue_tracker,
)
from vetinari.integrations.github_issues import GitHubIssueTracker
from vetinari.integrations.github_issues import _parse_issue as gh_parse
from vetinari.integrations.jira_adapter import JiraIssueTracker
from vetinari.integrations.jira_adapter import _parse_issue as jira_parse
from vetinari.integrations.linear_adapter import LinearIssueTracker
from vetinari.integrations.linear_adapter import _parse_issue as linear_parse

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ok_response(json_data: Any, status_code: int = 200) -> MagicMock:
    """Build a mock requests.Response that succeeds with the given JSON."""
    resp = MagicMock(spec=requests.Response)
    resp.ok = True
    resp.status_code = status_code
    resp.json.return_value = json_data
    resp.text = ""
    return resp


def _error_response(status_code: int, text: str = "error") -> MagicMock:
    """Build a mock requests.Response that signals an error."""
    resp = MagicMock(spec=requests.Response)
    resp.ok = False
    resp.status_code = status_code
    resp.json.return_value = {}
    resp.text = text
    return resp


# ---------------------------------------------------------------------------
# Data structure tests
# ---------------------------------------------------------------------------


class TestIssueDataclass:
    """Verify the frozen Issue dataclass behaves correctly."""

    def test_issue_default_values(self) -> None:
        issue = Issue(id="1", title="Bug")
        assert issue.description == ""
        assert issue.priority == IssuePriority.MEDIUM
        assert issue.status == IssueStatus.OPEN
        assert issue.labels == []
        assert issue.assignee is None
        assert issue.url == ""
        assert issue.tracker_type == ""
        assert issue.raw_data == {}

    def test_issue_immutable(self) -> None:
        issue = Issue(id="1", title="Bug")
        with pytest.raises(AttributeError):
            issue.title = "Other"  # type: ignore[misc]

    def test_issue_repr_contains_id_and_title(self) -> None:
        issue = Issue(id="42", title="Crash on startup", status=IssueStatus.RESOLVED)
        text = repr(issue)
        assert "42" in text
        assert "Crash on startup" in text
        assert "resolved" in text


class TestCreateIssueRequest:
    """Verify CreateIssueRequest default fields."""

    def test_defaults(self) -> None:
        req = CreateIssueRequest(title="Feature request")
        assert req.description == ""
        assert req.priority == IssuePriority.MEDIUM
        assert req.labels == []
        assert req.assignee is None

    def test_explicit_fields(self) -> None:
        req = CreateIssueRequest(
            title="Critical bug",
            description="App crashes",
            priority=IssuePriority.CRITICAL,
            labels=["backend", "urgent"],
            assignee="alice",
        )
        assert req.priority == IssuePriority.CRITICAL
        assert req.labels == ["backend", "urgent"]
        assert req.assignee == "alice"


# ---------------------------------------------------------------------------
# Factory function tests
# ---------------------------------------------------------------------------


class TestCreateIssueTrackerFactory:
    """Verify create_issue_tracker returns correct adapter or None."""

    def test_returns_none_for_empty_config(self) -> None:
        assert create_issue_tracker({}) is None

    def test_returns_none_for_missing_type(self) -> None:
        assert create_issue_tracker({"issue_tracker": {}}) is None

    def test_returns_none_for_unknown_type(self) -> None:
        assert create_issue_tracker({"issue_tracker": {"type": "gitlab"}}) is None

    def test_returns_none_for_incomplete_github_config(self) -> None:
        cfg = {"issue_tracker": {"type": "github", "github": {"token": "tok", "owner": ""}}}
        assert create_issue_tracker(cfg) is None

    def test_returns_none_for_incomplete_linear_config(self) -> None:
        cfg = {"issue_tracker": {"type": "linear", "linear": {"api_key": "key", "team_id": ""}}}
        assert create_issue_tracker(cfg) is None

    def test_returns_none_for_incomplete_jira_config(self) -> None:
        cfg = {
            "issue_tracker": {
                "type": "jira",
                "jira": {"url": "https://x.atlassian.net", "email": "", "api_token": "", "project_key": ""},
            }
        }
        assert create_issue_tracker(cfg) is None

    def test_returns_github_tracker_for_valid_config(self) -> None:
        cfg = {
            "issue_tracker": {
                "type": "github",
                "github": {"token": "ghp_abc", "owner": "acme", "repo": "myapp"},
            }
        }
        tracker = create_issue_tracker(cfg)
        assert isinstance(tracker, GitHubIssueTracker)

    def test_returns_linear_tracker_for_valid_config(self) -> None:
        cfg = {
            "issue_tracker": {
                "type": "linear",
                "linear": {"api_key": "lin_api_abc", "team_id": "team-uuid"},
            }
        }
        tracker = create_issue_tracker(cfg)
        assert isinstance(tracker, LinearIssueTracker)

    def test_returns_jira_tracker_for_valid_config(self) -> None:
        cfg = {
            "issue_tracker": {
                "type": "jira",
                "jira": {
                    "url": "https://company.atlassian.net",
                    "email": "dev@company.com",
                    "api_token": "tok",
                    "project_key": "VET",
                },
            }
        }
        tracker = create_issue_tracker(cfg)
        assert isinstance(tracker, JiraIssueTracker)


# ---------------------------------------------------------------------------
# GitHub adapter tests
# ---------------------------------------------------------------------------

_GH_RAW_ISSUE: dict[str, Any] = {
    "number": 7,
    "title": "Fix login",
    "body": "Login fails on mobile.",
    "state": "open",
    "html_url": "https://github.com/acme/app/issues/7",
    "labels": [{"name": "priority: high"}, {"name": "bug"}],
    "assignee": {"login": "bob"},
    "pull_request": None,  # not present means it's a real issue
}

# Remove pull_request key to simulate real issue
_GH_RAW_ISSUE_CLEAN = {k: v for k, v in _GH_RAW_ISSUE.items() if k != "pull_request"}


class TestGitHubParseIssue:
    """Unit tests for the GitHub response parser."""

    def test_parses_basic_fields(self) -> None:
        issue = gh_parse(_GH_RAW_ISSUE_CLEAN, "acme", "app")
        assert issue.id == "7"
        assert issue.title == "Fix login"
        assert issue.description == "Login fails on mobile."
        assert issue.status == IssueStatus.OPEN
        assert issue.priority == IssuePriority.HIGH
        assert issue.assignee == "bob"
        assert issue.url == "https://github.com/acme/app/issues/7"
        assert issue.tracker_type == "github"
        assert "bug" in issue.labels

    def test_closed_state(self) -> None:
        raw = dict(_GH_RAW_ISSUE_CLEAN, state="closed")
        issue = gh_parse(raw, "acme", "app")
        assert issue.status == IssueStatus.CLOSED

    def test_no_priority_label_defaults_to_medium(self) -> None:
        raw = dict(_GH_RAW_ISSUE_CLEAN, labels=[{"name": "bug"}])
        issue = gh_parse(raw, "acme", "app")
        assert issue.priority == IssuePriority.MEDIUM

    def test_critical_priority_label(self) -> None:
        raw = dict(_GH_RAW_ISSUE_CLEAN, labels=[{"name": "priority: critical"}])
        issue = gh_parse(raw, "acme", "app")
        assert issue.priority == IssuePriority.CRITICAL

    def test_no_assignee(self) -> None:
        raw = dict(_GH_RAW_ISSUE_CLEAN, assignee=None)
        issue = gh_parse(raw, "acme", "app")
        assert issue.assignee is None


@pytest.fixture
def gh_tracker() -> GitHubIssueTracker:
    """GitHub tracker with a mocked session."""
    with patch("vetinari.integrations.github_issues.create_session") as mock_cs:
        mock_cs.return_value = MagicMock()
        tracker = GitHubIssueTracker(token="ghp_test", owner="acme", repo="app")
        tracker._session = mock_cs.return_value
        return tracker


class TestGitHubCreateIssue:
    def test_create_issue_success(self, gh_tracker: GitHubIssueTracker) -> None:
        created_raw = dict(_GH_RAW_ISSUE_CLEAN, number=10, title="New bug")
        gh_tracker._session.post.return_value = _ok_response(created_raw, 201)

        issue = gh_tracker.create_issue(CreateIssueRequest(title="New bug"))
        assert issue.id == "10"
        assert issue.title == "New bug"

    def test_create_issue_api_error_raises(self, gh_tracker: GitHubIssueTracker) -> None:
        gh_tracker._session.post.return_value = _error_response(422, "Validation failed")

        with pytest.raises(IssueTrackerError) as exc_info:
            gh_tracker.create_issue(CreateIssueRequest(title="Bad"))
        assert "422" in str(exc_info.value)

    def test_create_issue_connection_error_raises(self, gh_tracker: GitHubIssueTracker) -> None:
        gh_tracker._session.post.side_effect = requests.RequestException("timeout")

        with pytest.raises(IssueTrackerError) as exc_info:
            gh_tracker.create_issue(CreateIssueRequest(title="Test"))
        assert "connection failed" in str(exc_info.value).lower()


class TestGitHubGetIssue:
    def test_get_issue_found(self, gh_tracker: GitHubIssueTracker) -> None:
        gh_tracker._session.get.return_value = _ok_response(_GH_RAW_ISSUE_CLEAN)
        issue = gh_tracker.get_issue("7")
        assert issue is not None
        assert issue.id == "7"

    def test_get_issue_not_found_returns_none(self, gh_tracker: GitHubIssueTracker) -> None:
        gh_tracker._session.get.return_value = _error_response(404)
        gh_tracker._session.get.return_value.ok = False
        result = gh_tracker.get_issue("999")
        assert result is None

    def test_get_issue_server_error_raises(self, gh_tracker: GitHubIssueTracker) -> None:
        gh_tracker._session.get.return_value = _error_response(500, "Server error")
        with pytest.raises(IssueTrackerError):
            gh_tracker.get_issue("7")


class TestGitHubListIssues:
    def test_list_issues_returns_parsed(self, gh_tracker: GitHubIssueTracker) -> None:
        gh_tracker._session.get.return_value = _ok_response([_GH_RAW_ISSUE_CLEAN])
        issues = gh_tracker.list_issues()
        assert len(issues) == 1
        assert issues[0].id == "7"

    def test_list_issues_filters_pull_requests(self, gh_tracker: GitHubIssueTracker) -> None:
        pr = dict(_GH_RAW_ISSUE_CLEAN, number=8, pull_request={"url": "..."})
        gh_tracker._session.get.return_value = _ok_response([_GH_RAW_ISSUE_CLEAN, pr])
        issues = gh_tracker.list_issues()
        assert len(issues) == 1

    def test_list_issues_with_status_filter(self, gh_tracker: GitHubIssueTracker) -> None:
        gh_tracker._session.get.return_value = _ok_response([])
        issues = gh_tracker.list_issues(status=IssueStatus.CLOSED)
        assert issues == []
        call_kwargs = gh_tracker._session.get.call_args
        params = call_kwargs.kwargs.get("params", {})
        assert params.get("state") == "closed"


class TestGitHubUpdateStatus:
    def test_update_status_success(self, gh_tracker: GitHubIssueTracker) -> None:
        gh_tracker._session.patch.return_value = _ok_response({})
        result = gh_tracker.update_status("7", IssueStatus.CLOSED)
        assert result is True

    def test_update_status_not_found(self, gh_tracker: GitHubIssueTracker) -> None:
        gh_tracker._session.patch.return_value = _error_response(404)
        gh_tracker._session.patch.return_value.ok = False
        result = gh_tracker.update_status("999", IssueStatus.CLOSED)
        assert result is False


class TestGitHubAddComment:
    def test_add_comment_success(self, gh_tracker: GitHubIssueTracker) -> None:
        gh_tracker._session.post.return_value = _ok_response({}, 201)
        result = gh_tracker.add_comment("7", "Looks good")
        assert result is True

    def test_add_comment_not_found(self, gh_tracker: GitHubIssueTracker) -> None:
        gh_tracker._session.post.return_value = _error_response(404)
        gh_tracker._session.post.return_value.ok = False
        result = gh_tracker.add_comment("999", "Hi")
        assert result is False


# ---------------------------------------------------------------------------
# Linear adapter tests
# ---------------------------------------------------------------------------

_LINEAR_RAW_ISSUE: dict[str, Any] = {
    "id": "lin-abc-123",
    "title": "Dark mode glitch",
    "description": "Colors wrong in dark mode.",
    "priority": 2,
    "url": "https://linear.app/team/issue/lin-abc-123",
    "state": {"name": "In Progress", "type": "started"},
    "assignee": {"name": "Alice", "email": "alice@example.com"},
    "labels": {"nodes": [{"name": "frontend"}]},
}


class TestLinearParseIssue:
    def test_parses_basic_fields(self) -> None:
        issue = linear_parse(_LINEAR_RAW_ISSUE)
        assert issue.id == "lin-abc-123"
        assert issue.title == "Dark mode glitch"
        assert issue.description == "Colors wrong in dark mode."
        assert issue.priority == IssuePriority.HIGH
        assert issue.status == IssueStatus.IN_PROGRESS
        assert issue.assignee == "alice@example.com"
        assert "frontend" in issue.labels
        assert issue.tracker_type == "linear"

    def test_completed_state(self) -> None:
        raw = dict(_LINEAR_RAW_ISSUE, state={"name": "Done", "type": "completed"})
        issue = linear_parse(raw)
        assert issue.status == IssueStatus.RESOLVED

    def test_unknown_priority_defaults_to_medium(self) -> None:
        raw = dict(_LINEAR_RAW_ISSUE, priority=99)
        issue = linear_parse(raw)
        assert issue.priority == IssuePriority.MEDIUM


@pytest.fixture
def lin_tracker() -> LinearIssueTracker:
    """Linear tracker with a mocked session."""
    with patch("vetinari.integrations.linear_adapter.create_session") as mock_cs:
        mock_cs.return_value = MagicMock()
        tracker = LinearIssueTracker(api_key="lin_api_test", team_id="team-uuid")
        tracker._session = mock_cs.return_value
        return tracker


def _linear_gql_ok(data: dict[str, Any]) -> MagicMock:
    return _ok_response({"data": data})


class TestLinearCreateIssue:
    def test_create_issue_success(self, lin_tracker: LinearIssueTracker) -> None:
        lin_tracker._session.post.return_value = _linear_gql_ok({
            "issueCreate": {"success": True, "issue": _LINEAR_RAW_ISSUE}
        })
        issue = lin_tracker.create_issue(CreateIssueRequest(title="Dark mode glitch"))
        assert issue.id == "lin-abc-123"
        assert issue.title == "Dark mode glitch"

    def test_create_issue_failure_raises(self, lin_tracker: LinearIssueTracker) -> None:
        lin_tracker._session.post.return_value = _linear_gql_ok({"issueCreate": {"success": False, "issue": None}})
        with pytest.raises(IssueTrackerError) as exc_info:
            lin_tracker.create_issue(CreateIssueRequest(title="Test"))
        assert "success=false" in str(exc_info.value)

    def test_create_issue_graphql_error_raises(self, lin_tracker: LinearIssueTracker) -> None:
        lin_tracker._session.post.return_value = _ok_response({"errors": [{"message": "Unauthorized"}]})
        with pytest.raises(IssueTrackerError) as exc_info:
            lin_tracker.create_issue(CreateIssueRequest(title="Test"))
        assert "Unauthorized" in str(exc_info.value)


class TestLinearGetIssue:
    def test_get_issue_found(self, lin_tracker: LinearIssueTracker) -> None:
        lin_tracker._session.post.return_value = _linear_gql_ok({"issue": _LINEAR_RAW_ISSUE})
        issue = lin_tracker.get_issue("lin-abc-123")
        assert issue is not None
        assert issue.id == "lin-abc-123"

    def test_get_issue_not_found_returns_none(self, lin_tracker: LinearIssueTracker) -> None:
        lin_tracker._session.post.return_value = _linear_gql_ok({"issue": None})
        result = lin_tracker.get_issue("missing")
        assert result is None

    def test_get_issue_not_found_error_returns_none(self, lin_tracker: LinearIssueTracker) -> None:
        lin_tracker._session.post.return_value = _ok_response({"errors": [{"message": "Entity not found"}]})
        result = lin_tracker.get_issue("gone")
        assert result is None


class TestLinearListIssues:
    def test_list_issues_returns_all(self, lin_tracker: LinearIssueTracker) -> None:
        lin_tracker._session.post.return_value = _linear_gql_ok({"team": {"issues": {"nodes": [_LINEAR_RAW_ISSUE]}}})
        issues = lin_tracker.list_issues()
        assert len(issues) == 1
        assert issues[0].id == "lin-abc-123"

    def test_list_issues_filtered_by_status(self, lin_tracker: LinearIssueTracker) -> None:
        lin_tracker._session.post.return_value = _linear_gql_ok({"team": {"issues": {"nodes": [_LINEAR_RAW_ISSUE]}}})
        # IN_PROGRESS matches the fixture state, so it should be included
        issues = lin_tracker.list_issues(status=IssueStatus.IN_PROGRESS)
        assert len(issues) == 1

    def test_list_issues_filtered_out(self, lin_tracker: LinearIssueTracker) -> None:
        lin_tracker._session.post.return_value = _linear_gql_ok({"team": {"issues": {"nodes": [_LINEAR_RAW_ISSUE]}}})
        issues = lin_tracker.list_issues(status=IssueStatus.CLOSED)
        assert issues == []


class TestLinearAddComment:
    def test_add_comment_success(self, lin_tracker: LinearIssueTracker) -> None:
        lin_tracker._session.post.return_value = _linear_gql_ok({"commentCreate": {"success": True}})
        result = lin_tracker.add_comment("lin-abc-123", "LGTM")
        assert result is True

    def test_add_comment_not_found(self, lin_tracker: LinearIssueTracker) -> None:
        lin_tracker._session.post.return_value = _ok_response({"errors": [{"message": "Entity not found — issue"}]})
        result = lin_tracker.add_comment("missing", "Hi")
        assert result is False


# ---------------------------------------------------------------------------
# Jira adapter tests
# ---------------------------------------------------------------------------

_JIRA_RAW_ISSUE: dict[str, Any] = {
    "key": "VET-42",
    "self": "https://company.atlassian.net/rest/api/3/issue/VET-42",
    "fields": {
        "summary": "Login timeout",
        "description": {
            "type": "doc",
            "version": 1,
            "content": [
                {
                    "type": "paragraph",
                    "content": [{"type": "text", "text": "Users get logged out after 5 minutes."}],
                }
            ],
        },
        "status": {
            "name": "In Progress",
            "statusCategory": {"key": "indeterminate"},
        },
        "priority": {"name": "High"},
        "assignee": {"emailAddress": "dev@company.com", "displayName": "Dev User"},
        "labels": ["backend", "auth"],
    },
}


class TestJiraParseIssue:
    def test_parses_basic_fields(self) -> None:
        issue = jira_parse(_JIRA_RAW_ISSUE)
        assert issue.id == "VET-42"
        assert issue.title == "Login timeout"
        assert "Users get logged out" in issue.description
        assert issue.priority == IssuePriority.HIGH
        assert issue.status == IssueStatus.IN_PROGRESS
        assert issue.assignee == "dev@company.com"
        assert "backend" in issue.labels
        assert issue.tracker_type == "jira"
        assert "VET-42" in issue.url

    def test_done_status_category(self) -> None:
        raw = dict(_JIRA_RAW_ISSUE)
        raw["fields"] = dict(
            _JIRA_RAW_ISSUE["fields"],
            status={"name": "Done", "statusCategory": {"key": "done"}},
        )
        issue = jira_parse(raw)
        assert issue.status == IssueStatus.RESOLVED

    def test_unknown_priority_defaults_to_medium(self) -> None:
        raw = dict(_JIRA_RAW_ISSUE)
        raw["fields"] = dict(_JIRA_RAW_ISSUE["fields"], priority={"name": "Custom"})
        issue = jira_parse(raw)
        assert issue.priority == IssuePriority.MEDIUM

    def test_string_description_parsed(self) -> None:
        raw = dict(_JIRA_RAW_ISSUE)
        raw["fields"] = dict(_JIRA_RAW_ISSUE["fields"], description="Plain text desc")
        issue = jira_parse(raw)
        assert issue.description == "Plain text desc"

    def test_null_description(self) -> None:
        raw = dict(_JIRA_RAW_ISSUE)
        raw["fields"] = dict(_JIRA_RAW_ISSUE["fields"], description=None)
        issue = jira_parse(raw)
        assert issue.description == ""


@pytest.fixture
def jira_tracker() -> JiraIssueTracker:
    """Jira tracker with a mocked session."""
    with patch("vetinari.integrations.jira_adapter.create_session") as mock_cs:
        mock_cs.return_value = MagicMock()
        tracker = JiraIssueTracker(
            url="https://company.atlassian.net",
            email="dev@company.com",
            api_token="tok",
            project_key="VET",
        )
        tracker._session = mock_cs.return_value
        return tracker


class TestJiraCreateIssue:
    def test_create_issue_success(self, jira_tracker: JiraIssueTracker) -> None:
        # First POST returns the created key, then GET returns full issue
        jira_tracker._session.post.return_value = _ok_response({"key": "VET-42"}, 201)
        jira_tracker._session.get.return_value = _ok_response(_JIRA_RAW_ISSUE)

        issue = jira_tracker.create_issue(CreateIssueRequest(title="Login timeout"))
        assert issue.id == "VET-42"
        assert issue.title == "Login timeout"

    def test_create_issue_api_error_raises(self, jira_tracker: JiraIssueTracker) -> None:
        jira_tracker._session.post.return_value = _error_response(400, "Bad request")
        with pytest.raises(IssueTrackerError) as exc_info:
            jira_tracker.create_issue(CreateIssueRequest(title="Bad"))
        assert "400" in str(exc_info.value)

    def test_create_issue_connection_error_raises(self, jira_tracker: JiraIssueTracker) -> None:
        jira_tracker._session.post.side_effect = requests.RequestException("refused")
        with pytest.raises(IssueTrackerError) as exc_info:
            jira_tracker.create_issue(CreateIssueRequest(title="Test"))
        assert "connection failed" in str(exc_info.value).lower()


class TestJiraGetIssue:
    def test_get_issue_found(self, jira_tracker: JiraIssueTracker) -> None:
        jira_tracker._session.get.return_value = _ok_response(_JIRA_RAW_ISSUE)
        issue = jira_tracker.get_issue("VET-42")
        assert issue is not None
        assert issue.id == "VET-42"

    def test_get_issue_not_found_returns_none(self, jira_tracker: JiraIssueTracker) -> None:
        jira_tracker._session.get.return_value = _error_response(404)
        jira_tracker._session.get.return_value.ok = False
        result = jira_tracker.get_issue("VET-999")
        assert result is None

    def test_get_issue_server_error_raises(self, jira_tracker: JiraIssueTracker) -> None:
        jira_tracker._session.get.return_value = _error_response(500, "Server error")
        with pytest.raises(IssueTrackerError):
            jira_tracker.get_issue("VET-42")


class TestJiraListIssues:
    def test_list_issues_returns_parsed(self, jira_tracker: JiraIssueTracker) -> None:
        jira_tracker._session.get.return_value = _ok_response({"issues": [_JIRA_RAW_ISSUE]})
        issues = jira_tracker.list_issues()
        assert len(issues) == 1
        assert issues[0].id == "VET-42"

    def test_list_issues_with_status_filter_passes_jql(self, jira_tracker: JiraIssueTracker) -> None:
        jira_tracker._session.get.return_value = _ok_response({"issues": []})
        jira_tracker.list_issues(status=IssueStatus.OPEN)
        call_kwargs = jira_tracker._session.get.call_args
        params = call_kwargs.kwargs.get("params", {})
        assert "To Do" in params.get("jql", "")

    def test_list_issues_connection_error_raises(self, jira_tracker: JiraIssueTracker) -> None:
        jira_tracker._session.get.side_effect = requests.RequestException("refused")
        with pytest.raises(IssueTrackerError):
            jira_tracker.list_issues()


class TestJiraUpdateStatus:
    def test_update_status_success(self, jira_tracker: JiraIssueTracker) -> None:
        transitions = {
            "transitions": [
                {"id": "31", "to": {"statusCategory": {"key": "done"}}},
            ]
        }
        jira_tracker._session.get.return_value = _ok_response(transitions)
        jira_tracker._session.post.return_value = _ok_response({}, 204)
        result = jira_tracker.update_status("VET-42", IssueStatus.RESOLVED)
        assert result is True

    def test_update_status_not_found(self, jira_tracker: JiraIssueTracker) -> None:
        jira_tracker._session.get.return_value = _error_response(404)
        jira_tracker._session.get.return_value.ok = False
        result = jira_tracker.update_status("VET-999", IssueStatus.RESOLVED)
        assert result is False

    def test_update_status_no_matching_transition_raises(self, jira_tracker: JiraIssueTracker) -> None:
        jira_tracker._session.get.return_value = _ok_response({"transitions": []})
        with pytest.raises(IssueTrackerError) as exc_info:
            jira_tracker.update_status("VET-42", IssueStatus.RESOLVED)
        assert "workflow" in str(exc_info.value).lower()


class TestJiraAddComment:
    def test_add_comment_success(self, jira_tracker: JiraIssueTracker) -> None:
        jira_tracker._session.post.return_value = _ok_response({}, 201)
        result = jira_tracker.add_comment("VET-42", "Looks good")
        assert result is True

    def test_add_comment_not_found(self, jira_tracker: JiraIssueTracker) -> None:
        jira_tracker._session.post.return_value = _error_response(404)
        jira_tracker._session.post.return_value.ok = False
        result = jira_tracker.add_comment("VET-999", "Hi")
        assert result is False

    def test_add_comment_connection_error_raises(self, jira_tracker: JiraIssueTracker) -> None:
        jira_tracker._session.post.side_effect = requests.RequestException("timeout")
        with pytest.raises(IssueTrackerError):
            jira_tracker.add_comment("VET-42", "Hi")


# ---------------------------------------------------------------------------
# IssueTrackerError tests
# ---------------------------------------------------------------------------


class TestIssueTrackerError:
    def test_message_and_status_code(self) -> None:
        exc = IssueTrackerError("API failed", status_code=503)
        assert str(exc) == "API failed"
        assert exc.status_code == 503

    def test_no_status_code(self) -> None:
        exc = IssueTrackerError("Connection refused")
        assert exc.status_code is None


# ---------------------------------------------------------------------------
# IssuePriority and IssueStatus enum tests
# ---------------------------------------------------------------------------


class TestEnums:
    @pytest.mark.parametrize(
        ("priority", "expected_value"),
        [
            (IssuePriority.CRITICAL, "critical"),
            (IssuePriority.HIGH, "high"),
            (IssuePriority.MEDIUM, "medium"),
            (IssuePriority.LOW, "low"),
        ],
    )
    def test_priority_values(self, priority: IssuePriority, expected_value: str) -> None:
        assert priority.value == expected_value

    @pytest.mark.parametrize(
        ("status", "expected_value"),
        [
            (IssueStatus.OPEN, "open"),
            (IssueStatus.IN_PROGRESS, "in_progress"),
            (IssueStatus.RESOLVED, "resolved"),
            (IssueStatus.CLOSED, "closed"),
        ],
    )
    def test_status_values(self, status: IssueStatus, expected_value: str) -> None:
        assert status.value == expected_value
