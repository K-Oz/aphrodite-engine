---
title: Troubleshooting Guide with Architecture Insights
description: Comprehensive troubleshooting guide with architectural context for common Aphrodite Engine issues
---

# Troubleshooting Guide with Architecture Insights

This guide helps you diagnose and resolve common issues with Aphrodite Engine by understanding the underlying architecture and component interactions.

## System Architecture for Troubleshooting

Understanding the request flow helps identify where issues might occur:

```mermaid
graph TB
    subgraph "Request Flow"
        A[Client Request] --> B[API Gateway]
        B --> C[Input Validation]
        C --> D[Request Scheduling]
        D --> E[Model Execution]
        E --> F[Token Generation]
        F --> G[Response Formatting]
        G --> H[Client Response]
    end
    
    subgraph "Common Failure Points"
        I[Network Issues<br/>Connection Timeout]
        J[Memory Issues<br/>OOM, Cache Full]
        K[GPU Issues<br/>CUDA Errors]
        L[Model Issues<br/>Loading Failures]
        M[Configuration Issues<br/>Invalid Parameters]
    end
    
    B -.-> I
    D -.-> J
    E -.-> K
    E -.-> L
    C -.-> M
```

## Memory-Related Issues

### Out of Memory (OOM) Errors

Memory issues are the most common problems in LLM inference:

```mermaid
graph TB
    subgraph "Memory Architecture"
        subgraph "GPU Memory Layout"
            ModelWeights[Model Weights<br/>~70% of memory]
            KVCache[KV Cache<br/>~20-25% of memory]
            TempBuffers[Temporary Buffers<br/>~5-10% of memory]
        end
        
        subgraph "Memory Pressure Points"
            LargeModel[Large Model Size]
            LongSequences[Long Context Length]
            LargeBatch[Large Batch Size]
            MultiModal[Multi-modal Inputs]
        end
    end
    
    subgraph "Solutions"
        Quantization[Model Quantization<br/>AWQ, GPTQ, FP8]
        BatchReduction[Reduce Batch Size]
        SequenceLimit[Limit Max Length]
        MemoryTuning[Tune Memory Utilization]
    end
    
    LargeModel --> Quantization
    LongSequences --> SequenceLimit
    LargeBatch --> BatchReduction
    MultiModal --> MemoryTuning
```

**Diagnostic Commands:**
```bash
# Check GPU memory usage
nvidia-smi

# Monitor Aphrodite memory usage
aphrodite run model_name --gpu-memory-utilization 0.8 --max-model-len 2048
```

**Solutions:**
1. **Reduce Memory Usage:**
   ```bash
   # Use quantization
   aphrodite run model_name --quantization awq
   
   # Reduce batch size
   aphrodite run model_name --max-num-seqs 32
   
   # Limit context length
   aphrodite run model_name --max-model-len 2048
   ```

2. **Use FP8 KV Cache:**
   ```bash
   aphrodite run model_name --kv-cache-dtype fp8
   ```

### KV Cache Management Issues

```mermaid
graph LR
    subgraph "KV Cache States"
        A[Empty Cache] --> B[Filling Cache]
        B --> C[Cache Full]
        C --> D[Eviction Policy]
        D --> E[Free Pages]
        E --> B
    end
    
    subgraph "Cache Issues"
        F[Cache Fragmentation]
        G[Memory Leaks]
        H[Invalid Page Access]
    end
    
    C -.-> F
    D -.-> G
    E -.-> H
```

**Diagnostic Signs:**
- Gradually increasing memory usage
- Sudden memory spikes
- Cache-related CUDA errors

**Solutions:**
```bash
# Enable prefix caching for efficiency
aphrodite run model_name --enable-prefix-caching

# Use block manager v2 for better memory management
aphrodite run model_name --use-v2-block-manager
```

## GPU and CUDA Issues

### CUDA Errors

```mermaid
graph TB
    subgraph "CUDA Error Categories"
        A[Memory Errors<br/>Out of memory, Invalid access]
        B[Kernel Errors<br/>Kernel launch failures]
        C[Driver Errors<br/>Driver version mismatch]
        D[Hardware Errors<br/>GPU hardware issues]
    end
    
    subgraph "Diagnostic Steps"
        E[Check CUDA Version]
        F[Verify Driver Version]
        G[Test GPU Memory]
        H[Monitor GPU Temperature]
    end
    
    subgraph "Solutions"
        I[Update Drivers]
        J[Reduce Memory Usage]
        K[Check Hardware]
        L[Restart CUDA Context]
    end
    
    A --> E
    B --> F
    C --> G
    D --> H
    
    E --> I
    F --> J
    G --> K
    H --> L
```

**Diagnostic Commands:**
```bash
# Check CUDA version
nvcc --version

# Check GPU status
nvidia-smi -q

# Test GPU memory
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_properties(0))"
```

### Multi-GPU Communication Issues

```mermaid
graph TB
    subgraph "Multi-GPU Architecture"
        GPU0[GPU 0] --> NCCL[NCCL Communication]
        GPU1[GPU 1] --> NCCL
        GPU2[GPU 2] --> NCCL
        GPU3[GPU 3] --> NCCL
    end
    
    subgraph "Communication Issues"
        A[NCCL Timeout]
        B[Bandwidth Bottleneck]
        C[Topology Issues]
        D[Version Mismatch]
    end
    
    subgraph "Diagnostics"
        E[Test NCCL]
        F[Check Topology]
        G[Monitor Bandwidth]
        H[Verify Versions]
    end
    
    NCCL -.-> A
    NCCL -.-> B
    NCCL -.-> C
    NCCL -.-> D
    
    A --> E
    B --> F
    C --> G
    D --> H
```

**Test NCCL Communication:**
```python
# Test PyTorch NCCL
import torch
import torch.distributed as dist
dist.init_process_group(backend="nccl")
local_rank = dist.get_rank() % torch.cuda.device_count()
torch.cuda.set_device(local_rank)
data = torch.FloatTensor([1,] * 128).to("cuda")
dist.all_reduce(data, op=dist.ReduceOp.SUM)
print("NCCL test successful!")
```

## Model Loading Issues

### Model Loading Flow

```mermaid
sequenceDiagram
    participant Client
    participant Engine
    participant ModelLoader
    participant Storage
    participant GPU
    
    Client->>Engine: Load Model Request
    Engine->>ModelLoader: Initialize Model
    ModelLoader->>Storage: Download/Load Weights
    Storage-->>ModelLoader: Model Files
    ModelLoader->>GPU: Transfer to GPU Memory
    GPU-->>ModelLoader: Memory Allocated
    ModelLoader-->>Engine: Model Ready
    Engine-->>Client: Engine Started
    
    Note over Storage: Common failure points:<br/>- Network issues<br/>- Disk space<br/>- Corrupted files
    Note over GPU: Common failure points:<br/>- Insufficient memory<br/>- CUDA errors<br/>- Driver issues
```

**Common Model Loading Issues:**

1. **Insufficient Disk Space:**
   ```bash
   # Check disk space
   df -h
   
   # Clean up cached models
   rm -rf ~/.cache/huggingface/hub/*
   ```

2. **Network Issues:**
   ```bash
   # Download model beforehand
   huggingface-cli download model_name
   
   # Use local path
   aphrodite run /path/to/local/model
   ```

3. **Permission Issues:**
   ```bash
   # Fix permissions
   sudo chown -R $USER:$USER ~/.cache/huggingface
   ```

## Performance Issues

### Latency Troubleshooting

```mermaid
graph TB
    subgraph "Latency Components"
        A[Network Latency<br/>Client ↔ Server]
        B[Queue Wait Time<br/>Request Scheduling]
        C[Model Execution<br/>Forward Pass]
        D[Token Generation<br/>Sampling & Decoding]
        E[Response Processing<br/>Formatting & Send]
    end
    
    subgraph "Optimization Strategies"
        F[Reduce Batch Size<br/>Lower Queue Time]
        G[Use Speculative Decoding<br/>Faster Generation]
        H[Enable Prefix Caching<br/>Faster Execution]
        I[Optimize Sampling<br/>Efficient Algorithms]
    end
    
    A --> F
    B --> F
    C --> G
    C --> H
    D --> I
    E --> I
```

**Latency Optimization:**
```bash
# Optimize for latency
aphrodite run model_name \
  --max-num-seqs 1 \
  --enable-prefix-caching \
  --speculative-model small_model \
  --enforce-eager
```

### Throughput Issues

```mermaid
graph TB
    subgraph "Throughput Bottlenecks"
        A[Small Batch Sizes]
        B[Memory Limitations]
        C[GPU Underutilization]
        D[I/O Bottlenecks]
    end
    
    subgraph "Solutions"
        E[Increase Batch Size]
        F[Optimize Memory Usage]
        G[Use Multiple GPUs]
        H[Async Processing]
    end
    
    A --> E
    B --> F
    C --> G
    D --> H
```

**Throughput Optimization:**
```bash
# Optimize for throughput
aphrodite run model_name \
  --max-num-seqs 256 \
  --max-num-batched-tokens 8192 \
  --tensor-parallel-size 4
```

## API and Networking Issues

### API Request Flow

```mermaid
sequenceDiagram
    participant Client
    participant LoadBalancer
    participant APIServer
    participant Engine
    participant Worker
    
    Client->>LoadBalancer: HTTP Request
    LoadBalancer->>APIServer: Route Request
    APIServer->>APIServer: Validate & Parse
    APIServer->>Engine: Process Request
    Engine->>Worker: Execute Model
    Worker-->>Engine: Generated Output
    Engine-->>APIServer: Response Data
    APIServer-->>LoadBalancer: HTTP Response
    LoadBalancer-->>Client: Final Response
    
    Note over Client,APIServer: Timeout issues<br/>Authentication failures<br/>Rate limiting
    Note over Engine,Worker: Processing errors<br/>Memory issues<br/>GPU errors
```

**Common API Issues:**

1. **Connection Timeouts:**
   ```bash
   # Increase timeout values
   aphrodite run model_name --timeout 300
   ```

2. **Rate Limiting:**
   ```bash
   # Adjust rate limits
   aphrodite run model_name --max-requests-per-minute 1000
   ```

3. **Authentication Issues:**
   ```bash
   # Set API keys
   aphrodite run model_name --api-keys "sk-key1,sk-key2"
   ```

## Monitoring and Diagnostics

### Health Check Architecture

```mermaid
graph TB
    subgraph "Health Monitoring"
        A[API Health Check<br/>/health endpoint]
        B[GPU Health Check<br/>nvidia-ml-py]
        C[Memory Health Check<br/>psutil monitoring]
        D[Model Health Check<br/>Inference test]
    end
    
    subgraph "Metrics Collection"
        E[Request Metrics<br/>Latency, Throughput]
        F[Resource Metrics<br/>GPU, Memory, CPU]
        G[Error Metrics<br/>Error rates, Types]
    end
    
    subgraph "Alerting"
        H[Threshold Alerts<br/>Performance degradation]
        I[Error Rate Alerts<br/>High error rates]
        J[Resource Alerts<br/>Memory/GPU limits]
    end
    
    A --> E
    B --> F
    C --> F
    D --> E
    
    E --> H
    F --> J
    G --> I
```

**Monitoring Commands:**
```bash
# Check API health
curl http://localhost:2242/health

# Monitor GPU usage
watch -n 1 nvidia-smi

# Monitor process resources
htop -p $(pgrep -f aphrodite)
```

## Advanced Debugging

### Debug Mode Configuration

```bash
# Enable debug logging
APHRODITE_LOG_LEVEL=DEBUG aphrodite run model_name

# Profile memory usage
APHRODITE_PROFILE_MEMORY=1 aphrodite run model_name

# Enable CUDA debugging
CUDA_LAUNCH_BLOCKING=1 aphrodite run model_name
```

### Debugging Distributed Setups

```mermaid
graph TB
    subgraph "Distributed Debugging"
        A[Network Connectivity<br/>Test inter-node communication]
        B[Ray Cluster Health<br/>Monitor Ray dashboard]
        C[NCCL Communication<br/>Test collective operations]
        D[Load Balancing<br/>Check request distribution]
    end
    
    subgraph "Diagnostic Tools"
        E[ray status<br/>Cluster overview]
        F[ray logs<br/>Distributed logs]
        G[nccl-tests<br/>Communication tests]
        H[tcpdump/wireshark<br/>Network analysis]
    end
    
    A --> H
    B --> E
    B --> F
    C --> G
    D --> H
```

This troubleshooting guide provides architectural context to help you understand and resolve issues more effectively by understanding how components interact within the Aphrodite Engine system.