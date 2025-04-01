# Agency Validation Report

*Generated on: 2025-04-01 02:26:01*

## Scenarios Tested

- **baseline**: Baseline model with no agency features activated
- **agency_default**: Agency model with all heads in active state
- **agency_mixed**: Agency model with mixed head states (30% overloaded, 20% misaligned)
- **agency_constrained**: Agency model with constrained resources (40% withdrawn)

## Performance Comparison

| Scenario | Tokens/sec | Relative Speed | Lexical Diversity | Repetition Score |
|----------|------------|----------------|-------------------|------------------|
| baseline | 23.68 | 100.00% | 0.759 | 0.023 |
| agency_default | 24.23 | 102.33% | 0.739 | 0.053 |
| agency_mixed | 24.35 | 102.83% | 0.778 | 0.015 |
| agency_constrained | 29.70 | 125.41% | 0.764 | 0.039 |

## Resource Usage

| Scenario | Generation Time (s) | CPU Usage (%) | RAM Usage (%) |
|----------|---------------------|---------------|---------------|
| baseline | 4.649 | 0.0 | 65.0 |
| agency_default | 4.498 | 87.4 | 64.6 |
| agency_mixed | 4.688 | 87.9 | 64.7 |
| agency_constrained | 4.027 | 86.8 | 64.7 |

## Agency States Distribution

| Scenario | Active Heads | Overloaded Heads | Misaligned Heads | Withdrawn Heads | Violations |
|----------|--------------|------------------|------------------|----------------|------------|
| baseline | 70 | 2 | 0 | 0 | 0 |
| agency_default | 70 | 2 | 0 | 0 | 0 |
| agency_mixed | 41 | 19 | 12 | 0 | 0 |
| agency_constrained | 47 | 1 | 0 | 24 | 20184 |

## Key Findings

- The best agency configuration (agency_constrained) was 25.4% faster than baseline
- Output quality was 0.7% higher than baseline

**Conclusion**: Agency features provide **SIGNIFICANT PERFORMANCE BENEFITS** with comparable quality

## Visualizations

- [generation_speed.png](generation_speed.png)
- [head_state_distribution.png](head_state_distribution.png)
- [generation_time.png](generation_time.png)
- [repetition_score_comparison.png](repetition_score_comparison.png)
- [lexical_diversity_comparison.png](lexical_diversity_comparison.png)
- [unique_token_ratio_comparison.png](unique_token_ratio_comparison.png)
