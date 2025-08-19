---
title: Deployment Architecture Guide
description: Comprehensive guide for deploying Aphrodite Engine in different architectures
---

# Deployment Architecture Guide

This guide covers various deployment architectures for Aphrodite Engine, from single-node setups to large-scale distributed deployments.

## Deployment Patterns

### Single Node Deployment

Perfect for development, testing, and small-scale production workloads:

```mermaid
graph TB
    subgraph "Single Node"
        subgraph "API Layer"
            API[Aphrodite API Server<br/>Port 2242]
        end
        
        subgraph "Engine Layer"
            Engine[Aphrodite Engine]
            Scheduler[Request Scheduler]
        end
        
        subgraph "Execution Layer"
            Worker1[GPU Worker 0]
            Worker2[GPU Worker 1]
            Worker3[GPU Worker 2]
            Worker4[GPU Worker 3]
        end
        
        subgraph "Hardware"
            GPU1[RTX 4090<br/>24GB VRAM]
            GPU2[RTX 4090<br/>24GB VRAM]
            GPU3[RTX 4090<br/>24GB VRAM]
            GPU4[RTX 4090<br/>24GB VRAM]
        end
    end
    
    Client[Client Applications] --> API
    API --> Engine
    Engine --> Scheduler
    Scheduler --> Worker1
    Scheduler --> Worker2
    Scheduler --> Worker3
    Scheduler --> Worker4
    
    Worker1 --> GPU1
    Worker2 --> GPU2
    Worker3 --> GPU3
    Worker4 --> GPU4
```

**Command:**
```bash
aphrodite run meta-llama/Meta-Llama-3.1-70B-Instruct \
  --tensor-parallel-size 4 \
  --gpu-memory-utilization 0.9 \
  --max-model-len 4096
```

### Multi-Node Cluster

For large models and high-throughput requirements:

```mermaid
graph TB
    subgraph "Load Balancer"
        LB[Nginx/HAProxy<br/>Load Balancer]
    end
    
    subgraph "Node 1 - Head Node"
        API1[API Server]
        Engine1[Engine Controller]
        Worker1_1[GPU Worker 0]
        Worker1_2[GPU Worker 1]
        GPU1_1[A100 80GB]
        GPU1_2[A100 80GB]
    end
    
    subgraph "Node 2 - Worker Node"
        Worker2_1[GPU Worker 2]
        Worker2_2[GPU Worker 3]
        GPU2_1[A100 80GB]
        GPU2_2[A100 80GB]
    end
    
    subgraph "Node 3 - Worker Node"
        Worker3_1[GPU Worker 4]
        Worker3_2[GPU Worker 5]
        GPU3_1[A100 80GB]
        GPU3_2[A100 80GB]
    end
    
    subgraph "Node 4 - Worker Node"
        Worker4_1[GPU Worker 6]
        Worker4_2[GPU Worker 7]
        GPU4_1[A100 80GB]
        GPU4_2[A100 80GB]
    end
    
    Client --> LB
    LB --> API1
    API1 --> Engine1
    Engine1 --> Worker1_1
    Engine1 --> Worker1_2
    Engine1 -.-> Worker2_1
    Engine1 -.-> Worker2_2
    Engine1 -.-> Worker3_1
    Engine1 -.-> Worker3_2
    Engine1 -.-> Worker4_1
    Engine1 -.-> Worker4_2
    
    Worker1_1 --> GPU1_1
    Worker1_2 --> GPU1_2
    Worker2_1 --> GPU2_1
    Worker2_2 --> GPU2_2
    Worker3_1 --> GPU3_1
    Worker3_2 --> GPU3_2
    Worker4_1 --> GPU4_1
    Worker4_2 --> GPU4_2
```

**Head Node:**
```bash
# Start Ray cluster head
ray start --head --port=6379

# Start Aphrodite
aphrodite run meta-llama/Meta-Llama-3.1-405B-Instruct \
  --tensor-parallel-size 8 \
  --pipeline-parallel-size 1 \
  --distributed-executor-backend ray
```

**Worker Nodes:**
```bash
# Join Ray cluster
ray start --address="head_node_ip:6379"
```

## Performance Optimization Deployments

### Memory-Optimized Configuration

For scenarios with limited GPU memory:

```mermaid
graph LR
    subgraph "Memory Optimization Strategies"
        A[FP8 KV Cache<br/>50% Memory Reduction]
        B[Model Quantization<br/>INT4/AWQ/GPTQ]
        C[Gradient Checkpointing<br/>Activation Memory]
        D[CPU Offloading<br/>Overflow to CPU]
    end
    
    subgraph "Configuration"
        Config[--kv-cache-dtype fp8<br/>--quantization awq<br/>--cpu-offload-gb 32<br/>--gpu-memory-utilization 0.95]
    end
    
    A --> Config
    B --> Config
    C --> Config
    D --> Config
```

### Throughput-Optimized Configuration

For maximum tokens per second:

```mermaid
graph LR
    subgraph "Throughput Optimization"
        A[Large Batch Sizes<br/>Continuous Batching]
        B[Multiple Engine Instances<br/>Process-level Parallelism]
        C[Tensor Parallelism<br/>Model Sharding]
        D[Flash Attention<br/>Memory-efficient Attention]
    end
    
    subgraph "Configuration"
        Config[--max-num-seqs 256<br/>--max-num-batched-tokens 8192<br/>--tensor-parallel-size 4<br/>--use-v2-block-manager]
    end
    
    A --> Config
    B --> Config
    C --> Config
    D --> Config
```

This deployment guide provides comprehensive patterns for running Aphrodite Engine across different environments and scales, ensuring optimal performance and reliability for your specific use case.