# Validation Results

This directory contains validation results and reports for Sentinel-AI experiments and features.

## Contents

- **agency/**: Validation results for attention head agency features
  - `sample_results.md`: Sample results showing expected validation metrics

## Running Validation

To run validation experiments, use the following scripts:

```bash
# For agency validation
python scripts/run_agency_validation.py --output validation_results/agency/
```

This will run all defined test scenarios and generate a comprehensive report including:
- Performance metrics
- Resource utilization measurements
- Output quality assessments
- Head state analysis
- Visualizations comparing all scenarios

## Interpreting Results

The validation reports include:

1. **Summary**: Overview of key findings and improvements
2. **Performance Metrics**: Generation speed and throughput measurements
3. **Resource Utilization**: Memory and computational resource usage
4. **Output Quality**: Perplexity, diversity, and reference-based metrics
5. **Head State Analysis**: Distribution of head states and transition patterns
6. **Detailed Analysis**: In-depth examination of specific scenarios
7. **Conclusions**: Overall assessment and recommendations

## Custom Validations

To create custom validation scenarios:

1. Edit `scripts/validate_agency.py` to define new test scenarios
2. Configure parameters in `scripts/run_agency_validation.py`
3. Run the validation script with your custom parameters
4. Results will be generated in your specified output directory