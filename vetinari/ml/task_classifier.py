"""TF-IDF + logistic regression task classifier — replaces LLM goal classification.

Classifies user task descriptions into GoalCategory values. When scikit-learn
is unavailable or fewer than MIN_TRAINING_EXAMPLES samples have been collected,
falls back to keyword matching (same logic as GoalClassifier in classifiers.py).

This is the primary classifier: LLM is only consulted as fallback when
model confidence < CONFIDENCE_THRESHOLD.

Pipeline role: Called by classify_goal_via_llm() in llm_helpers.py before
any LLM call is made.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Minimum labeled examples required before the trained model is used instead of keywords
MIN_TRAINING_EXAMPLES = 100

# Confidence below this threshold triggers LLM fallback
CONFIDENCE_THRESHOLD = 0.6

# Saved model location relative to project root
_MODEL_DIR = Path(".vetinari") / "models"
_MODEL_PATH = _MODEL_DIR / "task_classifier.pkl"


class TaskClassifier:
    """Classifies task descriptions into GoalCategory values.

    Uses TF-IDF + LogisticRegression when at least MIN_TRAINING_EXAMPLES
    labeled examples exist and scikit-learn is available.  Falls back to
    the keyword-matching GoalClassifier otherwise.

    Thread-safe: the internal sklearn model is protected by a lock and
    loaded/trained lazily on the first classify() call.

    Example::

        clf = TaskClassifier()
        category, confidence = clf.classify("implement a binary search tree")
        # -> ("code", 0.87)
    """

    def __init__(self) -> None:
        # sklearn objects; None until lazy-loaded
        self._model: Any = None
        self._vectorizer: Any = None
        self._training_examples: list[tuple[str, str]] = []
        self._lock = threading.Lock()
        self._sklearn_available: bool | None = None  # None = not yet probed
        self._keyword_classifier: Any = None  # cached GoalClassifier for keyword fallback

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def classify(self, text: str) -> tuple[str, float]:
        """Classify a task description into a GoalCategory.

        Checks whether the trained model should be used (enough examples and
        sklearn available).  Falls back to keyword matching otherwise.

        Args:
            text: The task description to classify.

        Returns:
            Tuple of (category_string, confidence_score_0_to_1).
        """
        if not text or not text.strip():
            return "general", 0.3

        if self._should_use_trained_model():
            result = self._classify_with_model(text)
            if result is not None:
                return result

        return self._classify_with_keywords(text)

    def add_example(self, text: str, label: str) -> None:
        """Add a labeled training example.

        When the total reaches MIN_TRAINING_EXAMPLES, the next classify()
        call will train and cache the sklearn model automatically.

        Args:
            text: The task description text.
            label: The GoalCategory value string (e.g. "code", "research").
        """
        with self._lock:
            self._training_examples.append((text, label))
        logger.debug(
            "TaskClassifier: added example (label=%r, total=%d)",
            label,
            len(self._training_examples),
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _should_use_trained_model(self) -> bool:
        """Return True when sklearn is available and a usable model is ready.

        A model is considered usable when either:
        - It is already loaded in memory (self._model is not None), or
        - Enough in-memory training examples exist to train a new one.

        A persisted model file on disk is NOT sufficient on its own because the
        file may be corrupt, incompatible, or left over from a previous run that
        never trained successfully.  The caller will attempt to load from disk
        inside ``_classify_with_model``; this method only gates the fast-path
        decision to bypass the keyword fallback entirely.
        """
        if self._sklearn_available is None:
            try:
                import sklearn  # noqa: F401 - import intentionally probes or re-exports API surface

                self._sklearn_available = True
            except ImportError:
                self._sklearn_available = False
                logger.debug("TaskClassifier: scikit-learn not installed — using keyword fallback")

        if not self._sklearn_available:
            return False

        # Only bypass keyword fallback when a model is already in memory or
        # we have enough examples to train one.  A stale disk file alone is not
        # enough — the load attempt may fail, and if training also fails (too
        # few examples) we would return nothing useful.
        return self._model is not None or len(self._training_examples) >= MIN_TRAINING_EXAMPLES

    def _classify_with_model(self, text: str) -> tuple[str, float] | None:
        """Classify using the trained sklearn model, training it if necessary.

        Args:
            text: The task description to classify.

        Returns:
            Tuple of (category, confidence), or None if training/inference fails.
        """
        with self._lock:
            # Try loading from disk first; if that fails, train from scratch
            if self._model is None and not self._try_load_from_disk() and not self._train_model():
                return None

        try:
            features = self._vectorizer.transform([text])
            probas = self._model.predict_proba(features)[0]
            best_idx = int(probas.argmax())
            confidence = float(probas[best_idx])
            category = self._model.classes_[best_idx]
            logger.debug(
                "TaskClassifier: ML result category=%r confidence=%.3f",
                category,
                confidence,
            )
            return str(category), confidence
        except Exception as exc:
            logger.warning(
                "TaskClassifier: ML inference failed (%s) — falling back to keywords",
                exc,
            )
            return None

    def _train_model(self) -> bool:
        """Train TF-IDF + LogisticRegression on current examples.

        Must be called inside self._lock.

        Returns:
            True if training succeeded, False otherwise.
        """
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.linear_model import LogisticRegression

            texts = [ex[0] for ex in self._training_examples]
            labels = [ex[1] for ex in self._training_examples]

            vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=10_000)
            features = vectorizer.fit_transform(texts)

            model = LogisticRegression(max_iter=1000, C=1.0)
            model.fit(features, labels)

            self._vectorizer = vectorizer
            self._model = model

            logger.info(
                "TaskClassifier: trained on %d examples (%d classes)",
                len(texts),
                len(set(labels)),
            )
            self._save_to_disk()
            return True
        except Exception as exc:
            logger.warning("TaskClassifier: training failed: %s", exc)
            return False

    def _try_load_from_disk(self) -> bool:
        """Load previously saved model from disk.

        Must be called inside self._lock.

        Returns:
            True if a saved model was loaded successfully.
        """
        if not _MODEL_PATH.exists():
            return False
        try:
            import joblib

            payload = joblib.load(_MODEL_PATH)  # noqa: VET302 — Trusted: loaded from project .vetinari/, not user-provided
            self._model = payload["model"]
            self._vectorizer = payload["vectorizer"]
            logger.info("TaskClassifier: loaded saved model from %s", _MODEL_PATH)
            return True
        except Exception as exc:
            logger.warning("TaskClassifier: could not load saved model: %s", exc)
            return False

    def _save_to_disk(self) -> None:
        """Persist the trained model to disk for reuse across process restarts.

        Must be called inside self._lock.
        """
        try:
            import joblib

            _MODEL_DIR.mkdir(parents=True, exist_ok=True)
            joblib.dump(
                {"model": self._model, "vectorizer": self._vectorizer},
                _MODEL_PATH,
            )
            logger.info("TaskClassifier: saved model to %s", _MODEL_PATH)
        except Exception as exc:
            logger.warning("TaskClassifier: could not save model: %s", exc)

    def _classify_with_keywords(self, text: str) -> tuple[str, float]:
        """Fall back to keyword-based GoalClassifier.

        The GoalClassifier instance is created once and reused — it holds no
        per-call state so caching it is safe and avoids repeated object allocation.

        Args:
            text: The task description to classify.

        Returns:
            Tuple of (category, confidence) from the keyword classifier.
        """
        if self._keyword_classifier is None:
            from vetinari.ml.classifiers import GoalClassifier

            self._keyword_classifier = GoalClassifier()
        result = self._keyword_classifier.classify(text)
        return result.category, result.confidence
