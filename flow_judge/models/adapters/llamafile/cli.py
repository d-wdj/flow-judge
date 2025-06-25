"""Command-line interface for managing Llamafile deployments."""

import argparse
import logging
import sys

from .server import LlamafileServer


def setup_logging(level: str = "INFO") -> None:
    """Setup logging configuration.
    
    :param level: Logging level
    """
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )


def cmd_start_server(args: argparse.Namespace) -> None:
    """Start a Llamafile server.
    
    :param args: Command line arguments
    """
    server = LlamafileServer(
        cache_dir=args.cache_dir,
        host=args.host,
        port=args.port,
        gpu_layers=args.gpu_layers,
        context_size=args.context_size,
        thread_count=args.threads,
        batch_size=args.batch_size,
        max_concurrent_requests=args.max_requests,
        temperature=args.temperature,
        max_new_tokens=args.max_tokens,
        disable_kv_offload=args.disable_kv_offload,
        quantized_kv=not args.no_quantized_kv,
        flash_attn=not args.no_flash_attn,
    )

    try:
        server.start()
        logging.info(f"Server running at http://{args.host}:{args.port}")
        logging.info("Press Ctrl+C to stop the server")

        # Keep the server running
        while True:
            import time
            time.sleep(1)

    except KeyboardInterrupt:
        logging.info("Received interrupt signal, stopping server...")
        server.stop()
    except Exception as e:
        logging.error(f"Server error: {e}")
        server.stop()
        sys.exit(1)


def cmd_test_connection(args: argparse.Namespace) -> None:
    """Test connection to a Llamafile server.
    
    :param args: Command line arguments
    """
    from .adapter import LlamafileAPIAdapter
    
    base_url = f"http://{args.host}:{args.port}/v1"
    adapter = LlamafileAPIAdapter(base_url=base_url)
    
    if adapter.health_check():
        print(f"✅ Successfully connected to Llamafile server at {base_url}")
        
        # Test a simple generation
        try:
            response = adapter.fetch_response([
                {"role": "user", "content": "Hello, how are you?"}
            ])
            print(f"✅ Test generation successful: {response[:100]}...")
        except Exception as e:
            print(f"❌ Test generation failed: {e}")
    else:
        print(f"❌ Cannot connect to Llamafile server at {base_url}")
        sys.exit(1)


def cmd_migration_guide(args: argparse.Namespace) -> None:
    """Show migration guide from local to remote Llamafile.
    
    :param args: Command line arguments
    """
    print("""
🔄 Llamafile Migration Guide: From Local to Remote Deployment

BEFORE (Local Deployment):
```python
from flow_judge.models import Llamafile

# This runs the model locally on the same machine
model = Llamafile(
    generation_params={
        "temperature": 0.1,
        "max_new_tokens": 1000,
    }
)
```

AFTER (Remote Deployment):

1️⃣ Start a standalone Llamafile server:
```bash
# Option A: Using the CLI
flow-judge-llamafile start-server --host 0.0.0.0 --port 8085

# Option B: Using Python
from flow_judge.models.adapters.llamafile import LlamafileServer

server = LlamafileServer(host="0.0.0.0", port=8085)
server.start()
```

2️⃣ Connect to the remote server:
```python
from flow_judge.models.adapters.llamafile import RemoteLlamafile

# Connect to your deployed server
model = RemoteLlamafile(
    base_url="http://your-server:8085/v1",
    generation_params={
        "temperature": 0.1,
        "max_new_tokens": 1000,
    }
)
```

3️⃣ For async operations:
```python
from flow_judge.models.adapters.llamafile import AsyncRemoteLlamafile

model = AsyncRemoteLlamafile(
    base_url="http://your-server:8085/v1",
)

# Use async methods
response = await model._async_generate("Your prompt here")
```

4️⃣ For combined sync/async:
```python
from flow_judge.models.adapters.llamafile import CombinedRemoteLlamafile

model = CombinedRemoteLlamafile(
    base_url="http://your-server:8085/v1",
)

# Use either sync or async methods
sync_response = model._generate("Your prompt")
async_response = await model._async_generate("Your prompt")
```

🌟 Benefits of Remote Deployment:
- ✅ Scalable: Deploy model on powerful GPU servers
- ✅ Flexible: Multiple clients can connect to the same model
- ✅ Resource efficient: Separate compute from application logic
- ✅ Production ready: Better for containerized deployments

🔧 Docker Deployment Example:
```dockerfile
FROM ubuntu:22.04

# Install dependencies
RUN apt-get update && apt-get install -y python3 python3-pip

# Copy your application
COPY . /app
WORKDIR /app

# Install flow-judge
RUN pip3 install flow-judge

# Start the server
EXPOSE 8085
CMD ["python3", "-m", "flow_judge.models.adapters.llamafile.cli", "start-server", "--host", "0.0.0.0"]
```

🔍 Testing Your Setup:
```bash
# Test connection to your server
flow-judge-llamafile test-connection --host your-server --port 8085
```
    """)


def main() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Llamafile Deployment CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--log-level", default="INFO", help="Logging level")
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Start server command
    start_parser = subparsers.add_parser("start-server", help="Start a Llamafile server")
    start_parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    start_parser.add_argument("--port", type=int, default=8085, help="Port to bind to")
    start_parser.add_argument("--cache-dir", default="~/.cache/flow-judge", help="Cache directory")
    start_parser.add_argument("--gpu-layers", type=int, default=34, help="Number of GPU layers")
    start_parser.add_argument("--context-size", type=int, default=8192, help="Context size")
    start_parser.add_argument("--threads", type=int, help="Number of threads")
    start_parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    start_parser.add_argument("--max-requests", type=int, default=1, help="Max concurrent requests")
    start_parser.add_argument("--temperature", type=float, default=0.1, help="Sampling temperature")
    start_parser.add_argument("--max-tokens", type=int, default=1000, help="Max new tokens")
    start_parser.add_argument("--disable-kv-offload", action="store_true", help="Disable KV offload")
    start_parser.add_argument("--no-quantized-kv", action="store_true", help="Disable quantized KV")
    start_parser.add_argument("--no-flash-attn", action="store_true", help="Disable flash attention")
    
    # Test connection command
    test_parser = subparsers.add_parser("test-connection", help="Test connection to a server")
    test_parser.add_argument("--host", default="localhost", help="Server host")
    test_parser.add_argument("--port", type=int, default=8085, help="Server port")
    
    # Migration guide command
    subparsers.add_parser("migration-guide", help="Show migration guide")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    setup_logging(args.log_level)
    
    if args.command == "start-server":
        cmd_start_server(args)
    elif args.command == "test-connection":
        cmd_test_connection(args)
    elif args.command == "migration-guide":
        cmd_migration_guide(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
