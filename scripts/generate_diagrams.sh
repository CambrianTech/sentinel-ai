#!/bin/bash
# Script to generate diagrams from Mermaid syntax files

# Check if mermaid-cli is installed
if ! command -v mmdc &> /dev/null
then
    echo "mermaid-cli not found, installing..."
    npm install -g @mermaid-js/mermaid-cli
fi

# Create output directory
mkdir -p docs/assets/figures

# Source directory containing mermaid files
MERMAID_DIR="docs/improved_diagrams/mermaid"
OUTPUT_DIR="docs/assets/figures"

# Parse the main_architecture.md file to extract individual diagram blocks
echo "Extracting diagram blocks from main_architecture.md"

# Process Main Architecture Diagram
echo "Generating Main Architecture Diagram..."
grep -A 50 "# Main Architecture Diagram" $MERMAID_DIR/main_architecture.md | \
    sed -n '/```mermaid/,/```/p' > $OUTPUT_DIR/main_architecture.mmd
mmdc -i $OUTPUT_DIR/main_architecture.mmd -o $OUTPUT_DIR/main_architecture.svg -t default
mmdc -i $OUTPUT_DIR/main_architecture.mmd -o $OUTPUT_DIR/main_architecture.png -t default -b transparent

# Process Attention Head with Agency
echo "Generating Attention Head with Agency Diagram..."
grep -A 50 "# Attention Head with Agency States" $MERMAID_DIR/main_architecture.md | \
    sed -n '/```mermaid/,/```/p' > $OUTPUT_DIR/agency_head_diagram.mmd
mmdc -i $OUTPUT_DIR/agency_head_diagram.mmd -o $OUTPUT_DIR/agency_head_diagram.svg -t default
mmdc -i $OUTPUT_DIR/agency_head_diagram.mmd -o $OUTPUT_DIR/agency_head_diagram.png -t default -b transparent

# Process Hybrid Adapter Architecture
echo "Generating Hybrid Adapter Architecture Diagram..."
grep -A 50 "# Hybrid Adapter Architecture" $MERMAID_DIR/main_architecture.md | \
    sed -n '/```mermaid/,/```/p' > $OUTPUT_DIR/hybrid_adapter.mmd
mmdc -i $OUTPUT_DIR/hybrid_adapter.mmd -o $OUTPUT_DIR/hybrid_adapter.svg -t default
mmdc -i $OUTPUT_DIR/hybrid_adapter.mmd -o $OUTPUT_DIR/hybrid_adapter.png -t default -b transparent

# Process Enhanced Controller with Feedback System
echo "Generating Enhanced Controller Diagram..."
grep -A 50 "# Enhanced Controller with Feedback System" $MERMAID_DIR/main_architecture.md | \
    sed -n '/```mermaid/,/```/p' > $OUTPUT_DIR/controller_diagram.mmd
mmdc -i $OUTPUT_DIR/controller_diagram.mmd -o $OUTPUT_DIR/controller_diagram.svg -t default
mmdc -i $OUTPUT_DIR/controller_diagram.mmd -o $OUTPUT_DIR/controller_diagram.png -t default -b transparent

# Process Adaptive Transformer Block
echo "Generating Adaptive Transformer Block Diagram..."
grep -A 50 "# Adaptive Transformer Block" $MERMAID_DIR/main_architecture.md | \
    sed -n '/```mermaid/,/```/p' > $OUTPUT_DIR/transformer_block.mmd
mmdc -i $OUTPUT_DIR/transformer_block.mmd -o $OUTPUT_DIR/transformer_block.svg -t default
mmdc -i $OUTPUT_DIR/transformer_block.mmd -o $OUTPUT_DIR/transformer_block.png -t default -b transparent

# Process U-Net Skip Connections in Transformer
echo "Generating U-Net Architecture Diagram..."
grep -A 50 "# U-Net Skip Connections in Transformer" $MERMAID_DIR/main_architecture.md | \
    sed -n '/```mermaid/,/```/p' > $OUTPUT_DIR/unet_architecture.mmd
mmdc -i $OUTPUT_DIR/unet_architecture.mmd -o $OUTPUT_DIR/unet_architecture.svg -t default
mmdc -i $OUTPUT_DIR/unet_architecture.mmd -o $OUTPUT_DIR/unet_architecture.png -t default -b transparent

# Generate dark theme versions for presentation
echo "Generating dark theme versions for presentations..."
mmdc -i $OUTPUT_DIR/main_architecture.mmd -o $OUTPUT_DIR/main_architecture_dark.svg -t dark
mmdc -i $OUTPUT_DIR/main_architecture.mmd -o $OUTPUT_DIR/main_architecture_dark.png -t dark -b transparent

mmdc -i $OUTPUT_DIR/agency_head_diagram.mmd -o $OUTPUT_DIR/agency_head_diagram_dark.svg -t dark
mmdc -i $OUTPUT_DIR/agency_head_diagram.mmd -o $OUTPUT_DIR/agency_head_diagram_dark.png -t dark -b transparent

# Clean up temporary files
echo "Cleaning up temporary files..."
find $OUTPUT_DIR -name "*.mmd" -delete

echo "All diagrams generated successfully in $OUTPUT_DIR"