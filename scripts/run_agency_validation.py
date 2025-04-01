#!/usr/bin/env python
"""
Run validation of agency features and generate a summary report.

This script:
1. Runs the agency validation script with different scenarios 
2. Generates a summary report with key findings
3. Produces visualizations comparing different scenarios

Usage:
    python scripts/run_agency_validation.py
"""

import os
import sys
import argparse
import subprocess
import json
import glob
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime

def run_validation(model_name="distilgpt2", output_dir="./validation_results", scenarios=None):
    """Run the agency validation script with specified parameters."""
    if not scenarios:
        scenarios = ["baseline", "agency_default", "agency_mixed", "agency_constrained"]
    
    cmd = [
        "python", "scripts/validate_agency.py",
        "--model_name", model_name,
        "--scenarios"] + scenarios + [
        "--output_dir", output_dir,
        "--num_prompts", "5",
        "--verbose"
    ]
    
    print(f"Running validation with command: {' '.join(cmd)}")
    subprocess.run(cmd)
    
    return os.path.join(output_dir, "validation_results.json")

def generate_report(results_file, output_dir):
    """Generate a detailed report from validation results."""
    if not os.path.exists(results_file):
        print(f"Results file not found: {results_file}")
        return
    
    # Load the results
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    # Create a timestamp for the report
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    report_file = os.path.join(output_dir, f"agency_validation_report_{timestamp}.md")
    
    # Generate the report
    with open(report_file, 'w') as f:
        f.write("# Agency Validation Report\n\n")
        f.write(f"*Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n")
        
        # Summary of scenarios tested
        f.write("## Scenarios Tested\n\n")
        for scenario, data in results.items():
            f.write(f"- **{scenario}**: {data.get('description', 'No description')}\n")
        f.write("\n")
        
        # Performance comparison
        f.write("## Performance Comparison\n\n")
        f.write("| Scenario | Tokens/sec | Relative Speed | Lexical Diversity | Repetition Score |\n")
        f.write("|----------|------------|----------------|-------------------|------------------|\n")
        
        baseline_speed = None
        if "baseline" in results and "inference" in results["baseline"]:
            baseline_speed = results["baseline"]["inference"]["tokens_per_second"]
        
        for scenario, data in results.items():
            if "inference" in data and "quality" in data:
                speed = data["inference"]["tokens_per_second"]
                rel_speed = speed / baseline_speed if baseline_speed else "N/A"
                diversity = data["quality"].get("lexical_diversity", "N/A")
                repetition = data["quality"].get("repetition_score", "N/A")
                
                if isinstance(rel_speed, float):
                    rel_speed = f"{rel_speed:.2%}"
                
                f.write(f"| {scenario} | {speed:.2f} | {rel_speed} | {diversity:.3f} | {repetition:.3f} |\n")
        
        f.write("\n")
        
        # Resource usage
        f.write("## Resource Usage\n\n")
        f.write("| Scenario | Generation Time (s) | CPU Usage (%) | RAM Usage (%) |\n")
        f.write("|----------|---------------------|---------------|---------------|\n")
        
        for scenario, data in results.items():
            if "resource_usage" in data:
                gen_time = data["resource_usage"].get("generation_time", "N/A")
                cpu = data["resource_usage"].get("cpu_percent", "N/A")
                ram = data["resource_usage"].get("ram_percent", "N/A")
                
                f.write(f"| {scenario} | {gen_time:.3f} | {cpu:.1f} | {ram:.1f} |\n")
        
        f.write("\n")
        
        # Agency report
        f.write("## Agency States Distribution\n\n")
        f.write("| Scenario | Active Heads | Overloaded Heads | Misaligned Heads | Withdrawn Heads | Violations |\n")
        f.write("|----------|--------------|------------------|------------------|----------------|------------|\n")
        
        for scenario, data in results.items():
            if "resource_usage" in data and "agency_report" in data["resource_usage"]:
                report = data["resource_usage"]["agency_report"]
                active = report.get("active_heads", "N/A")
                overloaded = report.get("overloaded_heads", "N/A")
                misaligned = report.get("misaligned_heads", "N/A")
                withdrawn = report.get("withdrawn_heads", "N/A")
                violations = report.get("total_violations", "N/A")
                
                f.write(f"| {scenario} | {active} | {overloaded} | {misaligned} | {withdrawn} | {violations} |\n")
        
        f.write("\n")
        
        # Key findings
        f.write("## Key Findings\n\n")
        
        # Compare agency scenarios to baseline
        if "baseline" in results and any(s != "baseline" for s in results.keys()):
            baseline_metrics = {
                "speed": results["baseline"]["inference"]["tokens_per_second"],
                "quality": results["baseline"]["quality"].get("lexical_diversity", 0),
            }
            
            best_agency_scenario = None
            best_agency_speed = 0
            
            for scenario, data in results.items():
                if scenario != "baseline" and data.get("agency_enabled", False):
                    speed = data["inference"]["tokens_per_second"]
                    if speed > best_agency_speed:
                        best_agency_speed = speed
                        best_agency_scenario = scenario
            
            if best_agency_scenario:
                speed_diff = best_agency_speed / baseline_metrics["speed"] - 1
                direction = "faster" if speed_diff > 0 else "slower"
                
                f.write(f"- The best agency configuration ({best_agency_scenario}) was {abs(speed_diff):.1%} {direction} than baseline\n")
                
                quality_diff = results[best_agency_scenario]["quality"].get("lexical_diversity", 0) / baseline_metrics["quality"] - 1
                quality_dir = "higher" if quality_diff > 0 else "lower"
                
                f.write(f"- Output quality was {abs(quality_diff):.1%} {quality_dir} than baseline\n")
                
                if speed_diff > 0 and quality_diff >= -0.05:  # Allow slight quality reduction
                    f.write("\n**Conclusion**: Agency features provide **SIGNIFICANT PERFORMANCE BENEFITS** with comparable quality\n")
                elif speed_diff > 0 and quality_diff < -0.05:
                    f.write("\n**Conclusion**: Agency features provide **PERFORMANCE BENEFITS** with some quality trade-offs\n")
                elif speed_diff <= 0 and quality_diff > 0:
                    f.write("\n**Conclusion**: Agency features provide **QUALITY BENEFITS** at some performance cost\n")
                else:
                    f.write("\n**Conclusion**: Agency benefits not clearly demonstrated in this configuration - further optimization needed\n")
        
        # Add links to visualizations
        f.write("\n## Visualizations\n\n")
        for img_file in glob.glob(os.path.join(output_dir, "*.png")):
            img_name = os.path.basename(img_file)
            f.write(f"- [{img_name}]({img_name})\n")
    
    print(f"Report generated: {report_file}")
    return report_file

def main():
    parser = argparse.ArgumentParser(description="Run agency validation and generate report")
    parser.add_argument("--model_name", type=str, default="distilgpt2", help="Model name")
    parser.add_argument("--output_dir", type=str, default="./validation_results/agency", 
                      help="Output directory")
    parser.add_argument("--scenarios", type=str, nargs="+", 
                      default=["baseline", "agency_default", "agency_mixed", "agency_constrained"],
                      help="Scenarios to test")
    parser.add_argument("--skip_validation", action="store_true", 
                      help="Skip running validation and just generate report from existing results")
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Run validation (unless skipped)
    results_file = os.path.join(args.output_dir, "validation_results.json")
    if not args.skip_validation:
        results_file = run_validation(
            model_name=args.model_name,
            output_dir=args.output_dir,
            scenarios=args.scenarios
        )
    
    # Generate report
    report_file = generate_report(results_file, args.output_dir)
    
    print(f"\nValidation complete.")
    print(f"Results: {results_file}")
    print(f"Report: {report_file}")

if __name__ == "__main__":
    main()