---
name: Experiment Runner
description: Design and run A/B tests with statistical rigor including hypothesis, methodology, success criteria, and analysis
mode: experiment
agent: worker
version: "1.0.0"
capabilities:
  - experiment_runner
tags:
  - operations
  - experiment
  - ab-test
  - statistics
---

# Experiment Runner

## Purpose

Experiment Runner designs, executes, and analyzes controlled experiments to validate hypotheses about system behavior, performance, or quality. It applies statistical rigor to ensure conclusions are reliable: explicit hypotheses, control and treatment groups, defined success criteria, appropriate sample sizes, and significance testing. This prevents the common failure mode of making changes based on anecdotal evidence or single observations, which leads to chasing noise rather than signal.

## When to Use

- When evaluating whether a proposed change actually improves a metric
- When comparing two approaches (model tiers, prompt strategies, decomposition methods)
- When the continuous improvement skill has identified a hypothesis to test
- When tuning system parameters (thinking budget, retry limits, cache TTL)
- When validating assumptions about system behavior with data
- When the team disagrees about the best approach and needs empirical evidence

## Inputs

| Parameter       | Type            | Required | Description                                                        |
|-----------------|-----------------|----------|--------------------------------------------------------------------|
| task            | string          | Yes      | Experiment objective and hypothesis                                |
| hypothesis      | string          | No       | Formal hypothesis: "X will improve Y by Z%"                       |
| control         | dict            | No       | Control group configuration (current behavior)                     |
| treatment       | dict            | No       | Treatment group configuration (proposed change)                    |
| metric          | string          | No       | Primary metric to measure (e.g., "task_success_rate")              |
| sample_size     | int             | No       | Minimum observations per group                                     |
| context         | dict            | No       | System context, constraints, baseline data                        |

## Process Steps

1. **Hypothesis formulation** -- State the hypothesis in falsifiable form:
   - Null hypothesis (H0): "There is no difference between control and treatment"
   - Alternative hypothesis (H1): "Treatment improves [metric] by at least [threshold]"
   - One-tailed vs two-tailed test selection based on the hypothesis direction

2. **Experiment design** -- Define the experimental setup:
   - **Independent variable**: what is being changed (model tier, prompt template, etc.)
   - **Dependent variable**: what is being measured (success rate, latency, cost)
   - **Control variables**: what must be held constant (input data, system load, etc.)
   - **Randomization**: how tasks are assigned to control vs treatment

3. **Sample size calculation** -- Determine minimum observations needed:
   - Desired statistical power (typically 80%)
   - Significance level (alpha, typically 0.05)
   - Minimum detectable effect size
   - Expected variance from historical data
   - Formula: n = (Z_alpha + Z_beta)^2 * 2 * sigma^2 / delta^2

4. **Baseline measurement** -- Measure the control group performance before starting treatment:
   - Collect baseline data for the primary metric
   - Record secondary metrics for side-effect detection
   - Verify baseline is stable (no trending, no anomalies)

5. **Treatment execution** -- Run the experiment:
   - Apply treatment to the treatment group only
   - Maintain identical conditions for the control group
   - Log all observations with timestamps and group labels
   - Monitor for early stopping conditions (dramatic failure, safety issues)

6. **Data collection** -- Gather results from both groups:
   - Primary metric values for each observation
   - Secondary metrics for side-effect analysis
   - Timestamps, durations, and any anomalies
   - Metadata that might explain variance (task complexity, file count)

7. **Statistical analysis** -- Analyze the results:
   - Descriptive statistics: mean, median, standard deviation for each group
   - Inferential statistics: t-test, chi-squared, or Mann-Whitney U as appropriate
   - P-value calculation and comparison against significance level
   - Effect size estimation (Cohen's d or relative improvement)
   - Confidence interval for the treatment effect

8. **Result interpretation** -- Draw conclusions:
   - Is the result statistically significant (p < alpha)?
   - Is the effect size practically meaningful (not just statistically significant)?
   - Are there confounding factors that could explain the result?
   - Are there side effects on secondary metrics?

9. **Decision and documentation** -- Based on results:
   - **Positive significant result**: recommend adopting the treatment
   - **Negative significant result**: recommend keeping the control (treatment hurts)
   - **Non-significant result**: inconclusive, recommend larger sample or different approach
   - Document everything for future reference and reproducibility

## Output Format

The skill produces an experiment report:

```json
{
  "success": true,
  "output": {
    "experiment_id": "EXP-2025-001",
    "hypothesis": "Using sonnet instead of opus for code_review will maintain quality while reducing cost by 40%",
    "design": {
      "independent_var": "model_tier for code_review mode",
      "dependent_var": "review_quality_score",
      "control": "opus tier",
      "treatment": "sonnet tier",
      "sample_size_per_group": 50,
      "randomization": "alternating task assignment"
    },
    "results": {
      "control": {"n": 50, "mean": 0.82, "std": 0.12, "median": 0.85},
      "treatment": {"n": 50, "mean": 0.79, "std": 0.14, "median": 0.81},
      "test": "two-sample t-test",
      "p_value": 0.23,
      "effect_size": -0.24,
      "significant": false,
      "confidence_interval": "[-0.09, +0.03] (95%)"
    },
    "conclusion": "No statistically significant difference in review quality between opus and sonnet (p=0.23). The 3-point mean difference is within noise. Recommend switching to sonnet for code_review to save ~40% cost.",
    "cost_impact": {
      "control_avg_cost": 0.12,
      "treatment_avg_cost": 0.07,
      "monthly_savings": 3.25
    },
    "side_effects": "None detected on secondary metrics (review length, issue count, false positive rate)"
  }
}
```

## Quality Standards

This skill is governed by the following standards from the skill registry:

- **STD-UNI-001**: All skill outputs MUST conform to the declared output_schema
- **STD-UNI-003**: All skill executions MUST log entry and exit at INFO level with timing
- **STD-UNI-004**: Skill executions SHOULD be idempotent -- running twice with same input produces same output
- **CON-WRK-009**: Operations modes run post-execution -- MUST NOT modify already-reviewed code
- **CON-UNI-002**: All skills must respect their max_tokens and timeout_seconds limits

## Examples

### Example: Testing a prompt optimization

**Input:**
```
task: "Test whether adding explicit examples to the build prompt improves first-attempt success rate"
hypothesis: "Adding 2 code examples to the build prompt will increase first-attempt pass rate from 70% to 85%"
metric: "first_attempt_pass_rate"
sample_size: 30
```

**Output (abbreviated):**
```
results:
  control: {n: 30, pass_rate: 0.70, passes: 21}
  treatment: {n: 30, pass_rate: 0.83, passes: 25}
  test: "chi-squared test for proportions"
  p_value: 0.048
  significant: true (at alpha=0.05)

conclusion: "Statistically significant improvement. Adding examples increased first-attempt pass rate from 70% to 83% (p=0.048). Effect is practically meaningful -- 4 fewer retries per 30 tasks."

side_effects:
  - "Input token count increased by 15% (examples add ~500 tokens per prompt)"
  - "Cost per build task increased by $0.008 but net savings from fewer retries is $0.02/task"

recommendation: "Adopt. Net cost reduction of $0.012 per task despite larger prompts."
```
