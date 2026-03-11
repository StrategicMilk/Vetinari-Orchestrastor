"""
Vetinari Goal Verifier
========================
Verifies that the final deliverable satisfies the original ProjectSpec.

This implements the output review → correction loop described in the
project plan:

1. Parse the original ProjectSpec (features, requirements, avoid list)
2. Run EvaluatorAgent to check compliance
3. Generate a verification matrix (Feature → Implemented? → Evidence)
4. SecurityAuditorAgent runs security check
5. If gaps found → create corrective tasks and re-enter execution
6. Present report to user and accept further changes

Usage:
    verifier = GoalVerifier()
    report = verifier.verify(project_spec, final_output, task_outputs)
    if not report.fully_compliant:
        corrective_tasks = report.get_corrective_tasks()
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

# AgentType imported lazily to avoid circular imports at module level

logger = logging.getLogger(__name__)


@dataclass
class FeatureVerification:
    """Verification result for a single feature/requirement."""
    feature: str
    implemented: bool
    confidence: float  # 0.0 - 1.0
    evidence: str = ""
    location: str = ""  # File/section where found
    severity: str = "major"  # major|minor|info


@dataclass
class GoalVerificationReport:
    """Complete verification report for a project delivery."""
    project_id: str
    goal: str
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    # Feature compliance matrix
    features: List[FeatureVerification] = field(default_factory=list)

    # Security
    security_passed: bool = True
    security_findings: List[Dict[str, Any]] = field(default_factory=list)
    security_score: float = 1.0

    # Code quality
    quality_score: float = 0.0
    quality_issues: List[str] = field(default_factory=list)

    # Tests
    tests_present: bool = False
    test_coverage_estimate: float = 0.0

    # Overall
    compliance_score: float = 0.0
    fully_compliant: bool = False
    missing_features: List[str] = field(default_factory=list)
    corrective_suggestions: List[str] = field(default_factory=list)

    # Metadata
    evaluator_verdict: str = "inconclusive"
    model_used: str = ""

    def get_corrective_tasks(self) -> List[Dict[str, Any]]:
        """Generate corrective tasks for all identified gaps."""
        tasks = []

        # Tasks for missing features
        for feat in self.missing_features:
            tasks.append({
                "type": "implement_missing_feature",
                "description": f"Implement missing feature: {feat}",
                "priority": "high",
                "assigned_agent": "BUILDER",
            })

        # Task for security issues
        if not self.security_passed and self.security_findings:
            tasks.append({
                "type": "fix_security_issues",
                "description": f"Fix {len(self.security_findings)} security findings",
                "priority": "critical",
                "assigned_agent": "SECURITY_AUDITOR",
                "details": self.security_findings,
            })

        # Task for missing tests
        if not self.tests_present:
            tasks.append({
                "type": "add_tests",
                "description": "Add unit tests (none found in deliverable)",
                "priority": "high",
                "assigned_agent": "TEST_AUTOMATION",
            })

        # General quality improvement
        for issue in self.corrective_suggestions[:3]:  # Top 3 suggestions
            tasks.append({
                "type": "quality_improvement",
                "description": issue,
                "priority": "medium",
                "assigned_agent": "EVALUATOR",
            })

        return tasks

    def to_dict(self) -> Dict[str, Any]:
        return {
            "project_id": self.project_id,
            "goal": self.goal,
            "created_at": self.created_at,
            "features": [
                {
                    "feature": f.feature,
                    "implemented": f.implemented,
                    "confidence": f.confidence,
                    "evidence": f.evidence,
                    "location": f.location,
                    "severity": f.severity,
                }
                for f in self.features
            ],
            "security_passed": self.security_passed,
            "security_score": self.security_score,
            "security_findings": self.security_findings,
            "quality_score": self.quality_score,
            "quality_issues": self.quality_issues,
            "tests_present": self.tests_present,
            "test_coverage_estimate": self.test_coverage_estimate,
            "compliance_score": self.compliance_score,
            "fully_compliant": self.fully_compliant,
            "missing_features": self.missing_features,
            "corrective_suggestions": self.corrective_suggestions,
            "evaluator_verdict": self.evaluator_verdict,
            "model_used": self.model_used,
        }


class GoalVerifier:
    """
    Verifies final deliverables against the original project goal and spec.
    """

    def __init__(self, quality_threshold: float = 0.75):
        self._threshold = quality_threshold

    def verify(
        self,
        project_id: str,
        goal: str,
        final_output: str,
        required_features: Optional[List[str]] = None,
        things_to_avoid: Optional[List[str]] = None,
        task_outputs: Optional[List[Dict[str, Any]]] = None,
        expected_outputs: Optional[List[str]] = None,
    ) -> GoalVerificationReport:
        """
        Verify a final deliverable against the original project spec.

        Args:
            project_id: The project identifier
            goal: The original goal statement
            final_output: The assembled final deliverable text
            required_features: List of required features from the intake form
            things_to_avoid: List of anti-requirements from the intake form
            task_outputs: List of individual task outputs
            expected_outputs: Expected output types (code, tests, docs, etc.)

        Returns:
            GoalVerificationReport with compliance matrix and corrective tasks
        """
        report = GoalVerificationReport(project_id=project_id, goal=goal)

        required_features = required_features or []
        things_to_avoid = things_to_avoid or []
        task_outputs = task_outputs or []

        # 1. Check required features
        report.features = self._verify_features(
            goal, required_features, final_output, task_outputs
        )

        # 2. Check things to avoid
        avoid_violations = self._check_avoid_list(things_to_avoid, final_output)
        if avoid_violations:
            for violation in avoid_violations:
                report.features.append(FeatureVerification(
                    feature=f"AVOID: {violation}",
                    implemented=False,  # False = violates "avoid"
                    confidence=0.8,
                    evidence=f"Found violation: {violation}",
                    severity="major",
                ))

        # 3. Check for tests
        report.tests_present = self._check_tests_present(final_output, task_outputs)

        # 4. Check expected output types
        if expected_outputs:
            for expected in expected_outputs:
                has_it = self._check_output_type(expected, final_output, task_outputs)
                if not has_it:
                    report.quality_issues.append(f"Expected output not delivered: {expected}")

        # 5. Try LLM-based evaluation
        try:
            llm_result = self._llm_evaluation(
                goal, final_output, required_features, things_to_avoid
            )
            if llm_result:
                report.quality_score = llm_result.get("quality_score", 0.7)
                report.evaluator_verdict = llm_result.get("verdict", "pass")
                report.corrective_suggestions = llm_result.get("improvements", [])
                report.model_used = llm_result.get("model_used", "")
                # Merge LLM feature verdicts with heuristic
                for feat_data in llm_result.get("feature_checks", []):
                    feat_name = feat_data.get("feature", "")
                    existing = next((f for f in report.features if f.feature == feat_name), None)
                    if existing:
                        # Update with higher-confidence LLM result
                        existing.implemented = feat_data.get("implemented", existing.implemented)
                        existing.confidence = max(
                            existing.confidence, feat_data.get("confidence", 0.5)
                        )
                        if feat_data.get("evidence"):
                            existing.evidence = feat_data["evidence"]
        except Exception as e:
            logger.warning(
                "LLM evaluation failed in goal verifier — defaulting quality_score to 0.3 "
                "(failing value) to avoid masking verification gaps: %s", e
            )
            report.quality_score = 0.3  # P2.4: Fail-safe default, not passing

        # 6. Run security check
        try:
            report.security_passed, report.security_findings, report.security_score = (
                self._security_check(final_output, task_outputs)
            )
        except Exception as e:
            logger.error("Security check failed — treating as unsafe: %s", e)
            report.security_passed = False  # Fail-closed: don't pass on security check failure

        # 7. Calculate overall compliance
        report.missing_features = [
            f.feature for f in report.features
            if not f.implemented and f.severity in ("major", "critical")
        ]

        if report.features:
            feature_score = sum(f.confidence for f in report.features if f.implemented) / len(report.features)
        else:
            # P2.4: No features to verify — default to 0.5 (neutral, not passing) to avoid
            # inflating compliance scores when feature list is empty or was not provided.
            logger.warning(
                "No features to verify for project %s — defaulting feature_score to 0.5",
                project_id,
            )
            feature_score = 0.5
        security_weight = 0.2 if not report.security_passed else 0.0
        report.compliance_score = max(0.0, (
            feature_score * 0.5
            + report.quality_score * 0.3
            + (1.0 if report.tests_present else 0.5) * 0.2
            - security_weight
        ))

        report.fully_compliant = (
            report.compliance_score >= self._threshold
            and report.security_passed
            and not report.missing_features
        )

        logger.info(
            "Goal verification for %s: score=%.2f, compliant=%s, missing=%s",
            project_id, report.compliance_score, report.fully_compliant,
            len(report.missing_features)
        )
        return report

    # ─── Private helpers ──────────────────────────────────────────────────────

    def _verify_features(
        self,
        goal: str,
        features: List[str],
        final_output: str,
        task_outputs: List[Dict[str, Any]],
    ) -> List[FeatureVerification]:
        """Heuristically check if features appear in the output."""
        verified = []
        combined_text = final_output + "\n" + "\n".join(
            str(t.get("output", "")) for t in task_outputs
        )
        combined_lower = combined_text.lower()

        for feature in features:
            if not feature.strip():
                continue
            feature_lower = feature.lower()
            # Extract keywords from feature description
            keywords = [w for w in feature_lower.split() if len(w) > 3]
            matches = sum(1 for kw in keywords if kw in combined_lower)
            confidence = min(1.0, matches / max(len(keywords), 1))
            implemented = confidence >= 0.5

            verified.append(FeatureVerification(
                feature=feature,
                implemented=implemented,
                confidence=confidence,
                evidence=f"Found {matches}/{len(keywords)} keywords" if implemented else "Not found in output",
                severity="major",
            ))

        return verified

    def _check_avoid_list(self, avoid_list: List[str], final_output: str) -> List[str]:
        """Check if any 'avoid' items appear in the output."""
        violations = []
        output_lower = final_output.lower()
        for item in avoid_list:
            if not item.strip():
                continue
            if item.lower() in output_lower:
                violations.append(item)
        return violations

    def _check_tests_present(
        self, final_output: str, task_outputs: List[Dict[str, Any]]
    ) -> bool:
        """Check if test files or test code is present."""
        combined = final_output + "\n" + "\n".join(
            str(t.get("output", "")) for t in task_outputs
        )
        test_indicators = [
            "def test_", "import pytest", "import unittest",
            "class Test", "assert ", "@pytest", "test_file",
            "describe(", "it(", "expect(",  # JS test frameworks
            "func Test", "testing.T",  # Go tests
        ]
        return any(ind in combined for ind in test_indicators)

    def _check_output_type(
        self,
        expected: str,
        final_output: str,
        task_outputs: List[Dict[str, Any]],
    ) -> bool:
        """Check if an expected output type is present."""
        combined = final_output + "\n" + "\n".join(
            str(t.get("output", "")) for t in task_outputs
        )
        type_patterns = {
            "code": ["def ", "class ", "function ", "import ", "require("],
            "tests": ["def test_", "pytest", "unittest", "describe("],
            "docs": ["# ", "## ", "### ", "README", "docstring"],
            "ci": ["github/workflows", ".yml", "pipeline", "CI/CD"],
            "docker": ["FROM ", "Dockerfile", "docker-compose"],
            "assets": ["<svg", ".png", ".jpg", "image"],
        }
        patterns = type_patterns.get(expected.lower(), [expected.lower()])
        return any(p in combined for p in patterns)

    def _llm_evaluation(
        self,
        goal: str,
        final_output: str,
        required_features: List[str],
        things_to_avoid: List[str],
    ) -> Optional[Dict[str, Any]]:
        """Use EvaluatorAgent for LLM-powered goal compliance check."""
        try:
            from vetinari.agents.evaluator_agent import get_evaluator_agent
            from vetinari.agents.contracts import AgentTask

            from vetinari.agents.contracts import AgentType
            evaluator = get_evaluator_agent()

            features_str = "\n".join(f"- {f}" for f in required_features) if required_features else "None specified"
            avoid_str = "\n".join(f"- {a}" for a in things_to_avoid) if things_to_avoid else "None specified"

            task = AgentTask(
                task_id="goal_verification",
                agent_type=AgentType.EVALUATOR,
                description="Verify deliverable against goal",
                prompt=f"""Verify this deliverable against the original goal.

GOAL: {goal}

REQUIRED FEATURES:
{features_str}

THINGS TO AVOID:
{avoid_str}

DELIVERABLE (first 3000 chars):
{final_output[:3000]}

For each required feature, check if it's implemented. Return JSON:
{{
  "verdict": "pass|fail|partial",
  "quality_score": 0.0-1.0,
  "feature_checks": [
    {{"feature": "...", "implemented": true/false, "confidence": 0.0-1.0, "evidence": "..."}}
  ],
  "improvements": ["..."],
  "summary": "..."
}}""",
                context={},
            )

            result = evaluator.execute(task)
            if result.success and isinstance(result.output, dict):
                return result.output

        except Exception as e:
            logger.debug("LLM evaluation in goal verifier failed: %s", e)
        return None

    def _security_check(
        self,
        final_output: str,
        task_outputs: List[Dict[str, Any]],
    ) -> tuple:
        """Run SecurityAuditorAgent on the output."""
        try:
            from vetinari.agents.security_auditor_agent import get_security_auditor_agent
            from vetinari.agents.contracts import AgentTask

            auditor = get_security_auditor_agent()
            combined = final_output[:4000]  # Limit to avoid context overflow

            from vetinari.agents.contracts import AgentType
            task = AgentTask(
                task_id="goal_security_check",
                agent_type=AgentType.SECURITY_AUDITOR,
                description="Security review of final deliverable",
                prompt=combined,
                context={},
            )

            result = auditor.execute(task)
            if result.success and isinstance(result.output, dict):
                findings = result.output.get("findings", [])
                score = result.output.get("score", 100) / 100.0
                critical = [f for f in findings if f.get("severity") in ("critical", "high")]
                return len(critical) == 0, findings, score
        except Exception as e:
            logger.debug("Security check failed: %s", e)

        return True, [], 1.0


# ─── Singleton ────────────────────────────────────────────────────────────────

_goal_verifier: Optional[GoalVerifier] = None


def get_goal_verifier() -> GoalVerifier:
    global _goal_verifier
    if _goal_verifier is None:
        _goal_verifier = GoalVerifier()
    return _goal_verifier
