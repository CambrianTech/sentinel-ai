# Main Architecture Diagram

```mermaid
flowchart TD
    classDef standard fill:#f9f9f9,stroke:#333,stroke-width:1px,color:#333
    classDef highlight fill:#d1f0ff,stroke:#0078b8,stroke-width:2px,color:#0078b8
    classDef adapter fill:#e6f7e6,stroke:#2e8b57,stroke-width:1px,color:#2e8b57
    classDef controller fill:#fff4e6,stroke:#ff8c00,stroke-width:1px,color:#ff8c00
    classDef attention fill:#f5e6ff,stroke:#9370db,stroke-width:1px,color:#9370db
    classDef embedding fill:#f9f9f9,stroke:#666,stroke-width:1px,color:#666
    
    %% Main Architecture Components
    title["SENTINEL-AI ARCHITECTURE"]
    adapterLayer["MODEL ADAPTER LAYER"]
    output["OUTPUT LAYER"]
    transformerBlocks["TRANSFORMER DECODER BLOCKS"]
    controller["ENHANCED CONTROLLER"]
    input["INPUT EMBEDDING"]
    
    %% Adapter Components
    gpt["GPT-2 Adapter"]
    bloom["BLOOM Adapter<br/>(ALiBi)"]
    llama["Llama Adapter<br/>(RoPE+SwiGLU)"]
    others["Other Adapters"]
    
    %% Transformer Block Components
    attention["Multi-Head Attention<br/>with Agency & Gates"]
    ffn["Feed Forward<br/>Network"]
    unet["U-Net Skip Connection"]
    
    %% Controller Components
    metrics["Metrics Collector"]
    rl["Reinforcement Learning<br/>Controller"]
    policy["Gate Update Policy"]
    reward["Reward Function<br/>(Performance Delta)"]
    
    %% Connections
    title --> adapterLayer
    
    adapterLayer --> gpt
    adapterLayer --> bloom
    adapterLayer --> llama
    adapterLayer --> others
    
    gpt & bloom & llama & others --> output
    
    output --> transformerBlocks
    
    transformerBlocks --> attention
    attention --> ffn
    
    transformerBlocks <--> unet
    
    transformerBlocks --> controller
    
    controller --> metrics
    controller --> rl
    controller --> policy
    metrics --> rl
    rl --> reward
    reward --> rl
    policy --> transformerBlocks
    
    controller --> input
    
    %% Styling
    title:::standard
    adapterLayer:::adapter
    output:::standard
    transformerBlocks:::highlight
    controller:::controller
    input:::embedding
    
    gpt & bloom & llama & others:::adapter
    
    attention & ffn:::attention
    unet:::highlight
    
    metrics & rl & policy & reward:::controller
```

# Attention Head with Agency States

```mermaid
flowchart TD
    classDef standard fill:#f9f9f9,stroke:#333,stroke-width:1px,color:#333
    classDef agency fill:#e6f7e6,stroke:#2e8b57,stroke-width:1px,color:#2e8b57
    classDef state fill:#f5e6ff,stroke:#9370db,stroke-width:1px,color:#9370db
    classDef computation fill:#d1f0ff,stroke:#0078b8,stroke-width:1px,color:#0078b8
    classDef gate fill:#fff4e6,stroke:#ff8c00,stroke-width:1px,color:#ff8c00
    
    %% Main Components
    title["ATTENTION HEAD WITH AGENCY"]
    signals["AGENCY SIGNALS"]
    stateProcessing["STATE PROCESSING"]
    monitor["CONSENT VIOLATION<br/>MONITORING"]
    attention["ATTENTION<br/>COMPUTATION"]
    gate["GATE MECHANISM<br/>output = gate_value * agency_factor * attn_out"]
    
    %% Agency Signal Components
    active["state: active<br/>consent: true/false<br/>utilization: 0.8<br/>last_signal: t"]
    
    %% State Components
    withdrawn["Withdrawn"]
    overloaded["Overloaded"]
    misaligned["Misaligned"]
    activeState["Active"]
    
    %% Action Components
    skipComputation["Skip Computation"]
    reduce50["Reduce Contribution<br/>by 50%"]
    reduce30["Reduce Contribution<br/>by 30%"]
    fullContribution["Full Contribution"]
    
    %% Connections
    title --> signals
    signals --> active
    
    signals --> stateProcessing
    signals --> monitor
    
    stateProcessing --> withdrawn & overloaded & misaligned & activeState
    
    withdrawn --> skipComputation
    overloaded --> reduce50
    misaligned --> reduce30
    activeState --> fullContribution
    
    skipComputation & reduce50 & reduce30 & fullContribution --> gate
    
    gate --> attention
    
    %% Styling
    title:::standard
    signals:::agency
    stateProcessing:::state
    monitor:::agency
    attention:::computation
    gate:::gate
    
    active:::agency
    
    withdrawn & overloaded & misaligned & activeState:::state
    
    skipComputation & reduce50 & reduce30 & fullContribution:::computation
```

# Hybrid Adapter Architecture

```mermaid
flowchart TD
    classDef standard fill:#f9f9f9,stroke:#333,stroke-width:1px,color:#333
    classDef interface fill:#d1f0ff,stroke:#0078b8,stroke-width:2px,color:#0078b8
    classDef adapter fill:#fff4e6,stroke:#ff8c00,stroke-width:1px,color:#ff8c00
    classDef original fill:#e6f7e6,stroke:#2e8b57,stroke-width:1px,color:#2e8b57
    
    %% Main Components
    title["HYBRID ADAPTER PATTERN"]
    interface["SENTINEL-AI INTERFACE"]
    adapter["MODEL-SPECIFIC ADAPTER"]
    original["ORIGINAL MODEL INTERNALS"]
    
    %% Adapter Components
    gates["DUMMY GATE LAYER"]
    compatible["CONTROLLER COMPATIBLE<br/>INTERFACE"]
    agency["AGENCY SIGNAL LAYER"]
    
    %% Original Model Components
    bloom["BLOOM: ALiBi Attention"]
    llama["LLAMA: Rotary Embeddings<br/>+ SwiGLU Activation"]
    
    %% Connections
    title --> interface
    interface --> adapter
    adapter --> gates & compatible & agency
    gates & compatible & agency --> original
    original --> bloom & llama
    
    %% Styling
    title:::standard
    interface:::interface
    adapter:::adapter
    original:::original
    
    gates & compatible & agency:::adapter
    
    bloom & llama:::original
```

# Enhanced Controller with Feedback System

```mermaid
flowchart TD
    classDef standard fill:#f9f9f9,stroke:#333,stroke-width:1px,color:#333
    classDef metrics fill:#e6f7e6,stroke:#2e8b57,stroke-width:1px,color:#2e8b57
    classDef reward fill:#fff4e6,stroke:#ff8c00,stroke-width:1px,color:#ff8c00
    classDef policy fill:#d1f0ff,stroke:#0078b8,stroke-width:1px,color:#0078b8
    classDef optimization fill:#f5e6ff,stroke:#9370db,stroke-width:1px,color:#9370db
    
    %% Main Components
    title["REINFORCEMENT LEARNING CONTROLLER"]
    metrics["VALIDATION METRICS<br/>COLLECTOR"]
    reward["REWARD CALCULATION<br/>reward = perf_improvement + efficiency_factor"]
    policy["POLICY NETWORK<br/>(Learns pruning patterns)"]
    history["ACTION HISTORY<br/>- Previous gate adjustments<br/>- State transitions<br/>- Reward history"]
    update["GATE VALUE UPDATE<br/>MECHANISM"]
    optimization["MULTI-OBJECTIVE<br/>OPTIMIZATION<br/>- Balance efficiency vs. performance<br/>- Task-specific specialization<br/>- Continuous adaptation"]
    
    %% Connections
    title --> metrics & policy & update & optimization
    
    metrics --> reward
    reward --> policy
    policy <--> history
    policy --> update
    update --> optimization
    
    %% Styling
    title:::standard
    metrics:::metrics
    reward:::reward
    policy:::policy
    history:::policy
    update:::policy
    optimization:::optimization
```

# Adaptive Transformer Block

```mermaid
flowchart TD
    classDef standard fill:#f9f9f9,stroke:#333,stroke-width:1px,color:#333
    classDef layer fill:#d1f0ff,stroke:#0078b8,stroke-width:1px,color:#0078b8
    classDef attention fill:#f5e6ff,stroke:#9370db,stroke-width:1px,color:#9370db
    classDef ffn fill:#e6f7e6,stroke:#2e8b57,stroke-width:1px,color:#2e8b57
    classDef head fill:#fff4e6,stroke:#ff8c00,stroke-width:1px,color:#ff8c00
    
    %% Main Components
    title["ADAPTIVE TRANSFORMER BLOCK"]
    residual["RESIDUAL CONNECTION"]
    norm["LAYER NORMALIZATION"]
    attention["MULTI-HEAD ATTENTION"]
    ffn["FEED FORWARD NETWORK"]
    dropout["DROPOUT"]
    output["OUTPUT"]
    
    %% Attention Components
    head1["HEAD 1<br/>+ AGENCY"]
    head2["HEAD 2<br/>+ AGENCY"]
    headn["HEAD N<br/>+ AGENCY"]
    
    gate1["GATE 1"]
    gate2["GATE 2"]
    gaten["GATE N"]
    
    %% Connections
    title --> residual & norm
    
    residual & norm --> attention
    
    attention --> head1 & head2 & headn
    head1 --> gate1
    head2 --> gate2
    headn --> gaten
    
    gate1 & gate2 & gaten --> ffn
    ffn --> dropout
    dropout --> output
    
    %% Styling
    title:::standard
    residual & norm:::layer
    attention:::attention
    ffn:::ffn
    dropout:::layer
    output:::layer
    
    head1 & head2 & headn:::attention
    gate1 & gate2 & gaten:::head
```

# U-Net Skip Connections in Transformer

```mermaid
flowchart TD
    classDef standard fill:#f9f9f9,stroke:#333,stroke-width:1px,color:#333
    classDef embedding fill:#d1f0ff,stroke:#0078b8,stroke-width:1px,color:#0078b8
    classDef decoder fill:#f5e6ff,stroke:#9370db,stroke-width:1px,color:#9370db
    classDef skip fill:#fff4e6,stroke:#ff8c00,stroke-width:1px,color:#ff8c00
    classDef encoder fill:#e6f7e6,stroke:#2e8b57,stroke-width:1px,color:#2e8b57
    
    %% Main Components
    title["U-NET INSPIRED ARCHITECTURE"]
    outputEmbed["OUTPUT EMBEDDING"]
    decoderBlocks["DECODER BLOCKS"]
    skipConnections["U-NET SKIP CONNECTIONS"]
    encoderBlocks["ENCODER BLOCKS"]
    inputEmbed["INPUT EMBEDDING"]
    
    %% Decoder Components
    blockN["Block N"]
    blockN1["Block N-1"]
    blockN2["Block N-2"]
    blockN3["Block N-3"]
    
    %% Skip Components
    fusion1["Fusion Function 1<br/>Linear([E;D])"]
    fusion2["Fusion Function 2<br/>Linear([E;D])"]
    fusion3["Fusion Function 3<br/>Linear([E;D])"]
    
    %% Encoder Components
    block1["Block 1"]
    block2["Block 2"]
    block3["Block 3"]
    
    %% Connections
    title --> outputEmbed
    outputEmbed --> decoderBlocks
    
    decoderBlocks --> blockN & blockN1 & blockN2 & blockN3
    
    blockN & blockN1 & blockN2 & blockN3 --> skipConnections
    
    skipConnections --> fusion1 & fusion2 & fusion3
    
    fusion1 & fusion2 & fusion3 --> encoderBlocks
    
    encoderBlocks --> block1 & block2 & block3
    
    block1 & block2 & block3 --> inputEmbed
    
    %% Styling
    title:::standard
    outputEmbed & inputEmbed:::embedding
    decoderBlocks:::decoder
    skipConnections:::skip
    encoderBlocks:::encoder
    
    blockN & blockN1 & blockN2 & blockN3:::decoder
    
    fusion1 & fusion2 & fusion3:::skip
    
    block1 & block2 & block3:::encoder
```