# Pruning System Architecture Flowchart

The following diagram illustrates the architecture and flow of the entropy and magnitude-based pruning system in Sentinel-AI.

```mermaid
flowchart TD
    subgraph Input
        Model[Model Architecture]
    end

    subgraph ArchitectureDetection[Architecture Detection]
        Model --> DetectArch[Detect Architecture]
        DetectArch --> GPT[GPT-style\nc_attn/c_proj]
        DetectArch --> BERT[BERT-style\nq/k/v_proj]
        DetectArch --> Adaptive[Sentinel-AI\nAdaptive Blocks]
    end

    subgraph EntropyPruning[Entropy-Based Pruning]
        GPT --> CollectAttn[Collect Attention\nDistributions]
        BERT --> CollectAttn
        Adaptive --> CollectBlocks[Collect Block\nAttention Maps]
        
        CollectAttn --> ComputeEntropy[Compute Entropy\nH = -∑p·log(p)]
        CollectBlocks --> ComputeEntropy
        
        ComputeEntropy --> RankEntropy[Rank Heads by\nEntropy (Highest First)]
    end

    subgraph MagnitudePruning[Magnitude-Based Pruning]
        GPT --> ExtractQKVO1[Extract Q/K/V/O\nWeights]
        BERT --> ExtractQKVO2[Extract Q/K/V/O\nWeights]
        Adaptive --> ExtractParams[Extract Block\nParameters]
        
        ExtractQKVO1 --> ComputeNorm1[Compute Weight Norms\nM = ||Q|| + ||K|| + ||V|| + ||O||]
        ExtractQKVO2 --> ComputeNorm2[Compute Weight Norms\nM = ||Q|| + ||K|| + ||V|| + ||O||]
        ExtractParams --> ComputeNorm3[Compute Parameter\nNorms]
        
        ComputeNorm1 --> RankMagnitude[Rank Heads by\nMagnitude (Lowest First)]
        ComputeNorm2 --> RankMagnitude
        ComputeNorm3 --> RankMagnitude
    end

    subgraph PruningApplication[Pruning Application]
        RankEntropy --> SelectHeads[Select Top N Heads\nBased on Prune Ratio]
        RankMagnitude --> SelectHeads
        
        SelectHeads --> FindMechanism{Determine Pruning\nMechanism}
        FindMechanism --> Gate[Gate\nMechanism]
        FindMechanism --> HeadMask[Head Mask\nMechanism]
        FindMechanism --> PruningMask[Pruning Mask\nMechanism]
        FindMechanism --> CreateMask[Create New\nMask Buffer]
        
        Gate --> SafeUpdate[Safe Tensor\nUpdate]
        HeadMask --> SafeUpdate
        PruningMask --> SafeUpdate
        CreateMask --> SafeUpdate
    end

    subgraph Output
        SafeUpdate --> PrunedModel[Pruned Model]
        PrunedModel --> FineTuning[Fine-Tuning]
        PrunedModel --> Evaluation[Evaluation]
        PrunedModel --> Visualization[Visualization]
    end

    %% Integration connections
    FineTuning -.-> |Metrics Collection| MetricsSystem[Metrics System]
    Evaluation -.-> |Perplexity, Generation| MetricsSystem
    Visualization -.-> |Gate Activity, Attention Maps| MetricsSystem

    %% Class styling
    classDef primary fill:#f9f,stroke:#333,stroke-width:2px;
    classDef secondary fill:#bbf,stroke:#333,stroke-width:1px;
    classDef tertiary fill:#dfd,stroke:#333,stroke-width:1px;
    
    class Model,PrunedModel primary;
    class CollectAttn,CollectBlocks,ExtractQKVO1,ExtractQKVO2,ExtractParams,ComputeEntropy,ComputeNorm1,ComputeNorm2,ComputeNorm3 secondary;
    class SelectHeads,SafeUpdate tertiary;
```

## Flowchart Description

The pruning system operates through these major stages:

### Architecture Detection
- The system first identifies the model architecture type (GPT, BERT, or Sentinel-AI Adaptive)
- This determines how attention mechanisms and weights are accessed

### Entropy-Based Pruning Path
1. **Attention Collection**: Gathers attention probability distributions via hooks
2. **Entropy Computation**: Calculates information-theoretic entropy for each head
3. **Head Ranking**: Sorts heads by entropy (highest entropy = least focused attention)

### Magnitude-Based Pruning Path
1. **Weight Extraction**: Extracts relevant weight matrices (Q/K/V/O) for each head
2. **Norm Computation**: Calculates weight norms to measure head importance 
3. **Head Ranking**: Sorts heads by magnitude (lowest magnitude = least important)

### Pruning Application
1. **Head Selection**: Selects heads to prune based on strategy and pruning ratio
2. **Mechanism Detection**: Identifies the appropriate pruning mechanism for the model
3. **Safe Update**: Applies pruning through gradient-preserving tensor updates

### Output Integration
- The pruned model integrates with fine-tuning, evaluation, and visualization systems
- Metrics are collected throughout the process for performance analysis and monitoring

## Notes

- The entropy and magnitude paths can operate independently or in combination
- Different model architectures use specialized extraction and computation methods
- The system is designed to gracefully handle unknown architectures with fallback mechanisms
- All pruning operations preserve gradient flow to ensure compatibility with subsequent training