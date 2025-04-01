# Sentinel-AI Validation Results

This directory contains validation results for various features of Sentinel-AI, with a focus on empirically demonstrating the benefits of our innovations.

## Agency Validation

The `agency/` directory contains validation results for our attention head agency features. These tests empirically measure the performance benefits of allowing attention heads to express internal states like "overloaded," "misaligned," or "withdrawn."

### Running Agency Validation

To run the agency validation yourself:

```bash
# Simple run with default settings
python scripts/run_agency_validation.py

# Specific model and scenarios
python scripts/run_agency_validation.py --model_name distilgpt2 --scenarios baseline agency_default agency_mixed

# Generate report from existing results
python scripts/run_agency_validation.py --skip_validation
```

### Understanding the Results

The validation compares different agency scenarios:

1. **baseline**: Standard model with no agency features activated
2. **agency_default**: All heads in active state with full consent
3. **agency_specialized**: Specialized head states for different tasks
4. **agency_mixed**: Mixed state configuration (some overloaded, some misaligned)
5. **agency_constrained**: Simulated resource constraints (some heads withdrawn)

For each scenario, we measure:
- **Generation speed** (tokens per second)
- **Output quality** (lexical diversity, repetition, etc.)
- **Resource utilization** (memory, CPU usage, etc.)
- **Agency state distribution** (active/overloaded/misaligned/withdrawn heads)

### Interpreting Visualizations

The validation produces several visualizations:

- **generation_speed.png**: Compares tokens per second across scenarios
- **lexical_diversity_comparison.png**: Compares output quality metrics
- **head_state_distribution.png**: Shows the distribution of head states in agency scenarios

## Other Validations

Additional validation results for other Sentinel-AI features may be added here in future updates.

## Reproducing Results

For consistent reproduction of validation results, use the same model, hardware configuration, and scenarios. Variations in hardware and model initialization may lead to slight differences in absolute metrics, but the relative performance patterns should remain consistent.

## Contributing New Validations

When adding new validation results, please include:
1. The script used to generate the results
2. A summary of key findings
3. Visualizations where appropriate
4. Details on the testing environment

This helps ensure that all results in this repository can be verified and reproduced.