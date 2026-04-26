---
name: Cost Analysis
description: Token-level cost analysis per model with tier optimization recommendations and budget projections
mode: cost_analysis
agent: worker
version: "1.0.0"
capabilities:
  - cost_analysis
tags:
  - operations
  - cost
  - budget
  - optimization
---

# Cost Analysis

## Purpose

Cost Analysis provides detailed token-level cost breakdowns per model tier, identifies optimization opportunities to reduce cost without sacrificing quality, and projects future costs based on usage trends. It enables informed decisions about model tier selection, thinking budget allocation, and task routing to balance cost against capability. For Vetinari, where every agent invocation has a token cost, understanding and optimizing cost is essential for sustainable operation.

## When to Use

- After completing a plan to understand the actual cost versus estimate
- When evaluating whether a task can be routed to a cheaper model tier
- When budgets are constrained and cost reduction is needed
- During capacity planning to project monthly costs
- When comparing the cost-effectiveness of different decomposition strategies
- When the Foreman needs cost data to inform effort estimation
- After a spike in costs to identify the cause

## Inputs

| Parameter       | Type            | Required | Description                                                        |
|-----------------|-----------------|----------|--------------------------------------------------------------------|
| task            | string          | Yes      | What to analyze and the analysis objective                         |
| usage_data      | list[dict]      | No       | Token usage records (model, input_tokens, output_tokens, mode)     |
| model_pricing   | dict            | No       | Per-model pricing (input_cost_per_1k, output_cost_per_1k)         |
| time_range      | dict            | No       | Analysis period: {start: "2025-01-01", end: "2025-01-31"}         |
| budget          | dict            | No       | Budget constraints (daily_limit, monthly_limit)                    |
| context         | dict            | No       | System context (task types, model configuration)                   |

## Process Steps

1. **Usage data collection** -- Gather token usage records from episode memory, execution logs, or provided usage data. For each invocation, record: model tier, input tokens, output tokens, mode, task type, and timestamp.

2. **Per-model cost calculation** -- Apply pricing to each invocation:
   - Input cost = input_tokens * input_price_per_token
   - Output cost = output_tokens * output_price_per_token
   - Total cost = input_cost + output_cost
   - Aggregate by model tier, mode, and task type

3. **Cost distribution analysis** -- Identify where costs concentrate:
   - Which model tier accounts for the most cost?
   - Which mode (build, research, review) is most expensive?
   - Which task types have the highest per-task cost?
   - What percentage of cost is input tokens vs output tokens?

4. **Efficiency metrics** -- Calculate efficiency ratios:
   - Cost per successful task completion
   - Cost per line of code produced (for build tasks)
   - Cost per review (for inspector tasks)
   - Retry cost (wasted tokens on failed attempts)
   - Token waste ratio (tokens spent on tasks that were later replanned)

5. **Tier optimization analysis** -- For each task type, evaluate if a cheaper model tier could handle it:
   - Research tasks: could use efficient tier instead of capable?
   - Simple build tasks: could reduce thinking budget?
   - Standard reviews: could use lighter review mode?
   - Estimate cost savings for each optimization

6. **Thinking budget analysis** -- Analyze thinking budget usage:
   - Distribution of thinking modes (low/medium/high/xhigh) by task type
   - Correlation between thinking budget and task success rate
   - Identify tasks where high thinking was used but low would suffice
   - Identify tasks where low thinking led to failures (under-invested)

7. **Trend projection** -- Based on historical data, project future costs:
   - Daily/weekly/monthly cost trends
   - Growth rate of token usage
   - Projected cost for the next period
   - When the budget will be exhausted at current rate

8. **Optimization recommendations** -- Produce actionable recommendations:
   - Model tier downgrade opportunities with risk assessment
   - Thinking budget adjustments with expected savings
   - Task batching opportunities (reduce per-invocation overhead)
   - Caching opportunities (avoid re-computing similar tasks)

9. **Report assembly** -- Compile the cost analysis report with: summary metrics, per-model breakdown, optimization opportunities, trend projections, and recommendations.

## Output Format

The skill produces a cost analysis report:

```json
{
  "success": true,
  "output": {
    "summary": {
      "total_cost_usd": 12.45,
      "period": "2025-01-01 to 2025-01-31",
      "total_invocations": 342,
      "avg_cost_per_task": 0.036
    },
    "by_model": {
      "claude-opus": {"invocations": 45, "cost": 8.20, "pct": 65.9},
      "claude-sonnet": {"invocations": 180, "cost": 3.50, "pct": 28.1},
      "claude-haiku": {"invocations": 117, "cost": 0.75, "pct": 6.0}
    },
    "by_mode": {
      "build": {"invocations": 89, "cost": 5.60},
      "code_discovery": {"invocations": 120, "cost": 2.10},
      "code_review": {"invocations": 65, "cost": 3.80},
      "plan": {"invocations": 68, "cost": 0.95}
    },
    "optimization_opportunities": [
      {
        "opportunity": "Route code_discovery tasks to haiku tier",
        "current_cost": 2.10,
        "projected_cost": 0.45,
        "savings_usd": 1.65,
        "risk": "low -- discovery tasks are pattern-matching, not reasoning-intensive"
      }
    ],
    "projection": {
      "next_month_estimate": 14.20,
      "growth_rate": "14% month-over-month",
      "budget_exhaustion": "Not at risk (budget: $50/month)"
    }
  },
  "metadata": {
    "data_source": "episode_memory",
    "records_analyzed": 342
  }
}
```

## Quality Standards

This skill is governed by the following standards from the skill registry:

- **STD-WRK-018**: Cost analysis MUST include per-model token breakdown and total estimated cost
- **STD-UNI-001**: All skill outputs MUST conform to the declared output_schema
- **STD-UNI-003**: All skill executions MUST log entry and exit at INFO level with timing
- **CON-WRK-008**: Cost analysis must not access external billing APIs without explicit permission
- **CON-WRK-009**: Operations modes run post-execution -- MUST NOT modify already-reviewed code

## Examples

### Example: Monthly cost review with optimization

**Input:**
```
task: "Analyze January costs and recommend optimizations to stay under $10/month"
time_range: {start: "2025-01-01", end: "2025-01-31"}
budget: {monthly_limit: 10.00}
```

**Output (abbreviated):**
```
summary: Total $12.45 (24.5% over budget)

top cost driver: code_review mode using opus tier ($3.80, 30.5% of total)

recommendations:
  1. [high] Route standard code_review to sonnet tier ($3.80 -> $1.50, saves $2.30)
     Risk: "Slight reduction in nuanced review quality for edge cases"

  2. [medium] Reduce thinking_mode for code_discovery from medium to low ($2.10 -> $1.40, saves $0.70)
     Risk: "May miss subtle patterns in unfamiliar code"

  3. [low] Cache prompt assembly results to reduce input tokens ($0.40 savings)
     Risk: "None -- pure optimization"

projected_with_optimizations: $8.85/month (within $10 budget)
```
