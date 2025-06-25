# Enhanced Llamafile Class - Remote Mode Support

This document demonstrates the new remote mode functionality added to the `Llamafile` class, which allows connecting to externally deployed llamafile servers instead of only running locally.

## Usage Examples

### Local Mode (Default Behavior - No Changes)

```python
from flow_judge.models import Llamafile

# Traditional local mode - starts llamafile server locally
judge = Llamafile(
    generation_params={
        "temperature": 0.1,
        "max_new_tokens": 1000,
    },
    gpu_layers=34,
    context_size=8192
)

# Use the judge for evaluation
result = judge.evaluate(inputs, outputs, criteria="accuracy")
```

### Remote Mode (New Feature)

```python
from flow_judge.models import Llamafile

# Remote mode - connects to external llamafile server
judge = Llamafile(
    remote=True,
    base_url="http://your-llamafile-server:8080/v1",
    api_key="your-api-key-if-needed",  # Optional
    generation_params={
        "temperature": 0.1,
        "max_new_tokens": 1000,
    },
    timeout=60,
    max_retries=3,
    retry_delay=1.0,
    max_concurrent_requests=10
)

# Use exactly the same as local mode
result = judge.evaluate(inputs, outputs, criteria="accuracy")
```

### Remote Mode with Authentication

```python
# If your external server requires authentication
judge = Llamafile(
    remote=True,
    base_url="https://secure-llamafile-api.example.com/v1",
    api_key="sk-your-secret-api-key",
    timeout=120,  # Longer timeout for remote calls
    max_retries=5
)
```

## Key Benefits

1. **Scalability**: Deploy llamafile on powerful GPU servers while running flow-judge on lightweight client machines
2. **Resource Efficiency**: Multiple flow-judge instances can share a single powerful llamafile deployment
3. **Flexibility**: Switch between local and remote modes with a single flag
4. **Backward Compatibility**: Existing code continues to work unchanged
5. **Error Handling**: Proper validation and error messages for remote configuration

## Configuration Parameters

### Remote Mode Parameters

- `remote: bool = False` - Enable remote mode
- `base_url: str` - Base URL of the external llamafile server (required when remote=True)
- `api_key: str = "not-needed"` - API key for authentication
- `timeout: int = 60` - Request timeout in seconds
- `max_retries: int = 3` - Maximum retry attempts for failed requests
- `retry_delay: float = 1.0` - Delay between retry attempts
- `max_concurrent_requests: int = 10` - Maximum concurrent requests to remote server

### Deployment Scenarios

#### Scenario 1: Local Development
```python
# Quick testing and development
judge = Llamafile()  # Uses default local mode
```

#### Scenario 2: Production with Dedicated GPU Server
```python
# Production setup with external llamafile server
judge = Llamafile(
    remote=True,
    base_url="http://gpu-server.internal:8080/v1",
    max_concurrent_requests=20,
    timeout=120
)
```

#### Scenario 3: Cloud API Service
```python
# Using a managed llamafile service
judge = Llamafile(
    remote=True,
    base_url="https://api.llamafile-service.com/v1",
    api_key="your-service-api-key",
    timeout=60,
    max_retries=5
)
```

## Error Handling

The enhanced Llamafile class provides proper error handling for remote mode:

```python
# This will raise ValueError
try:
    judge = Llamafile(remote=True)  # Missing base_url
except ValueError as e:
    print(f"Configuration error: {e}")

# This will raise LlamafileError if server is unreachable
try:
    judge = Llamafile(
        remote=True,
        base_url="http://unreachable-server:8080/v1"
    )
    result = judge.evaluate(...)  # Will check connectivity here
except LlamafileError as e:
    print(f"Server connectivity error: {e}")
```

## Migration Guide

### From Local to Remote

1. **Identify your llamafile server URL**: Deploy llamafile on your target server and note the URL
2. **Add remote parameters**: Update your Llamafile initialization with `remote=True` and `base_url`
3. **Adjust concurrency**: Consider increasing `max_concurrent_requests` for remote servers
4. **Test connectivity**: Verify the remote server is accessible before deploying

### Example Migration

```python
# Before (local mode)
judge = Llamafile(
    generation_params={"temperature": 0.1},
    gpu_layers=34
)

# After (remote mode)
judge = Llamafile(
    remote=True,
    base_url="http://your-server:8080/v1",
    generation_params={"temperature": 0.1},
    max_concurrent_requests=20  # Can be higher for remote servers
    # gpu_layers not needed in remote mode
)
```

The evaluation code remains exactly the same!
