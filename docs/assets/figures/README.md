# Figures Directory

This directory contains diagrams and figures for the Sentinel-AI documentation.

To generate high-quality diagrams from the Mermaid source files, use mermaid-cli:

```bash
# Install mermaid-cli
npm install -g @mermaid-js/mermaid-cli

# Generate SVG for each Mermaid diagram
mmdc -i ../../improved_diagrams/mermaid/main_architecture.md -o main_architecture.svg -t default

# Generate PNG with transparent background
mmdc -i ../../improved_diagrams/mermaid/main_architecture.md -o main_architecture.png -t default -b transparent

# Generate dark theme versions
mmdc -i ../../improved_diagrams/mermaid/main_architecture.md -o main_architecture_dark.svg -t dark
```

## Available Diagrams

1. **Main Architecture**: Complete system architecture including adapters and controller
2. **Attention Head with Agency**: Detailed view of agency mechanisms within attention heads
3. **Hybrid Adapter Architecture**: Structure of the hybrid adapter pattern for BLOOM and Llama models
4. **Enhanced Controller**: Architecture of the reinforcement learning controller
5. **Adaptive Transformer Block**: Detail view of a single adaptive transformer block
6. **U-Net Skip Connections**: Visualization of U-Net style skip connections in the transformer

## Usage in Documentation

When adding figures to documentation, use the following Markdown pattern:

```markdown
![Alt Text](../docs/assets/figures/diagram_name.png)
```

For the paper, use a similar pattern but with a different relative path:

```markdown
![Alt Text](./figures/diagram_name.png)
```