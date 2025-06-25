"""Standalone Llamafile server deployment utilities."""

import argparse
import atexit
import logging
import os
import shlex
import signal
import subprocess
import threading
import time
import weakref
from typing import Any

import requests
from tqdm import tqdm

LLAMAFILE_URL = (
    "https://huggingface.co/flowaicom/Flow-Judge-v0.1-Llamafile/resolve/main/"
    "flow-judge.llamafile"
)

logger = logging.getLogger(__name__)


class LlamafileServerError(Exception):
    """Custom exception for Llamafile server-related errors."""

    def __init__(self, status_code: int, message: str):
        """Initialize a LlamafileServerError with a status code and message."""
        self.status_code = status_code
        self.message = message
        super().__init__(self.message)


class DownloadError(Exception):
    """Exception raised when there's an error downloading the Llamafile."""

    pass


def cleanup_llamafile(process_ref):
    """Clean up the Llamafile process.

    :param process_ref: Weak reference to the Llamafile process.
    """
    process = process_ref()
    if process:
        pgid = None
        try:
            pgid = os.getpgid(process.pid)
            os.killpg(pgid, signal.SIGTERM)
            process.wait(timeout=5)
        except (subprocess.TimeoutExpired, ProcessLookupError):
            if pgid is not None:
                try:
                    os.killpg(pgid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
        except OSError:
            # Handle the case where we can't get the process group ID
            try:
                process.terminate()
                process.wait(timeout=5)
            except (subprocess.TimeoutExpired, ProcessLookupError):
                try:
                    process.kill()
                except ProcessLookupError:
                    pass


class LlamafileServer:
    """Standalone Llamafile server for external deployment."""

    def __init__(
        self,
        model_filename: str = "flow-judge.llamafile",
        cache_dir: str = os.path.expanduser("~/.cache/flow-judge"),
        host: str = "0.0.0.0",
        port: int = 8085,
        disable_kv_offload: bool = False,
        quantized_kv: bool = True,
        flash_attn: bool = True,
        context_size: int = 8192,
        gpu_layers: int = 34,
        thread_count: int | None = None,
        batch_size: int = 32,
        max_concurrent_requests: int = 1,
        temperature: float = 0.1,
        max_new_tokens: int = 1000,
        additional_args: dict[str, Any] | None = None,
        **kwargs: Any,
    ):
        """Initialize the standalone Llamafile server.

        :param model_filename: Name of the Llamafile model file
        :param cache_dir: Directory to cache the model files
        :param host: Host to bind the server to (0.0.0.0 for external access)
        :param port: Port to run the server on
        :param disable_kv_offload: Whether to disable KV offloading
        :param quantized_kv: Whether to enable Quantized KV
        :param flash_attn: Whether to enable Flash Attention
        :param context_size: Size of the context window
        :param gpu_layers: Number of GPU layers to use
        :param thread_count: Number of threads (None = auto-detect)
        :param batch_size: Batch size for processing
        :param max_concurrent_requests: Maximum number of concurrent requests
        :param temperature: Sampling temperature
        :param max_new_tokens: Maximum number of new tokens to generate
        :param additional_args: Additional command line arguments
        :param kwargs: Additional keyword arguments
        """
        self.model_filename = model_filename
        self.cache_dir = cache_dir
        self.host = host
        self.port = port
        self.disable_kv_offload = disable_kv_offload
        self.quantized_kv = quantized_kv
        self.flash_attn = flash_attn
        self.context_size = context_size
        self.gpu_layers = gpu_layers
        self.thread_count = thread_count or os.cpu_count()
        self.batch_size = batch_size
        self.max_concurrent_requests = max_concurrent_requests
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.additional_args = additional_args or {}

        self.process = None
        self._is_running = False

        if self.quantized_kv and not self.flash_attn:
            raise LlamafileServerError(
                status_code=1,
                message="Quantized KV is enabled but Flash Attention is disabled. "
                "This configuration won't function."
            )

    def download_llamafile(self) -> str:
        """Download the Llamafile model.

        :return: Path to the downloaded Llamafile.
        :raises DownloadError: If the download fails.
        """
        os.makedirs(self.cache_dir, exist_ok=True)
        llamafile_path = os.path.abspath(
            os.path.join(self.cache_dir, self.model_filename)
        )

        if not os.path.exists(llamafile_path):
            logger.info(f"Downloading llamafile to {llamafile_path}")
            try:
                response = requests.get(LLAMAFILE_URL, stream=True, timeout=30)
                response.raise_for_status()

                total_size = int(response.headers.get("content-length", 0))
                block_size = 8192

                with (
                    open(llamafile_path, "wb") as file,
                    tqdm(
                        desc="Downloading",
                        total=total_size,
                        unit="iB",
                        unit_scale=True,
                        unit_divisor=1024,
                    ) as progress_bar,
                ):
                    for data in response.iter_content(block_size):
                        size = file.write(data)
                        progress_bar.update(size)

            except requests.exceptions.RequestException as e:
                logger.error(f"Error downloading llamafile: {str(e)}")
                if os.path.exists(llamafile_path):
                    os.remove(llamafile_path)
                raise DownloadError(f"Failed to download llamafile: {str(e)}") from e

            except Exception as e:
                logger.error(f"Unexpected error during download: {str(e)}")
                if os.path.exists(llamafile_path):
                    os.remove(llamafile_path)
                raise DownloadError(f"Unexpected error during download: {str(e)}") from e

        try:
            os.chmod(llamafile_path, 0o755)
            logger.debug(f"Set executable permissions for {llamafile_path}")
        except OSError as e:
            logger.error(f"Failed to set executable permissions: {str(e)}")
            raise DownloadError(f"Failed to set executable permissions: {str(e)}") from e

        return llamafile_path

    def _build_command(self, llamafile_path: str) -> str:
        """Build the command to start the Llamafile server.

        :param llamafile_path: Path to the Llamafile executable
        :return: Command string to execute
        """
        command = (
            f"sh -c '{llamafile_path} --server --host {self.host} --port {self.port} "
            f"-c {self.context_size} "
            f"-ngl {self.gpu_layers} "
            f"--temp {self.temperature} "
            f"-n {self.max_new_tokens} "
            f"--threads {self.thread_count} "
            f"--nobrowser -b {self.batch_size} "
            f"--parallel {self.max_concurrent_requests} "
            f"--cont-batching"
        )

        if self.disable_kv_offload:
            command += " -nkvo"
            logger.info("KV offloading disabled")

        if self.quantized_kv:
            command += " -ctk q4_0 -ctv q4_0"
            logger.info("Quantized KV enabled")

        if self.flash_attn:
            command += " -fa"
            logger.info("Flash Attention enabled")

        # Add any additional arguments
        for key, value in self.additional_args.items():
            command += f" --{key} {value}"
            logger.info(f"Additional server argument added: --{key} {value}")

        command += "'"
        return command

    def _start_process(self, command: str) -> subprocess.Popen:
        """Start the Llamafile process.

        :param command: Command to execute
        :return: Started subprocess
        """
        def log_output(pipe, log_func):
            for line in iter(pipe.readline, ""):
                log_func(line.strip())

        process = subprocess.Popen(
            shlex.split(command),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=1,
            universal_newlines=True,
            text=True,
            preexec_fn=os.setsid,
        )
        logger.info(f"Subprocess started with PID: {process.pid}")

        # Register cleanup function for this specific process
        atexit.register(cleanup_llamafile, weakref.ref(process))

        # Start threads to log stdout and stderr in real-time
        threading.Thread(
            target=log_output, args=(process.stdout, logger.info), daemon=True
        ).start()
        threading.Thread(
            target=log_output, args=(process.stderr, logger.info), daemon=True
        ).start()

        return process

    def is_server_running(self) -> bool:
        """Check if the server is running and responding.

        :return: True if the server is running, False otherwise
        """
        try:
            response = requests.get(
                f"http://{self.host}:{self.port}/v1/models", timeout=5
            )
            response.raise_for_status()
            return True
        except requests.RequestException:
            return False

    def start(self, wait_for_ready: bool = True, timeout: int = 60) -> None:
        """Start the Llamafile server.

        :param wait_for_ready: Whether to wait for the server to be ready
        :param timeout: Timeout in seconds to wait for server startup
        :raises LlamafileServerError: If server fails to start
        """
        if self._is_running:
            logger.warning("Server is already running")
            return

        logger.info("Starting Llamafile server...")
        llamafile_path = self.download_llamafile()
        logger.info(f"Llamafile path: {llamafile_path}")

        if not os.path.exists(llamafile_path):
            raise LlamafileServerError(
                status_code=2, message=f"Llamafile not found at {llamafile_path}"
            )

        if not os.access(llamafile_path, os.X_OK):
            raise LlamafileServerError(
                status_code=3, message=f"Llamafile at {llamafile_path} is not executable"
            )

        command = self._build_command(llamafile_path)
        logger.info(f"Starting server with command: {command}")

        try:
            self.process = self._start_process(command)
            self._is_running = True

            if wait_for_ready:
                self._wait_for_ready(timeout)

        except Exception as e:
            logger.exception(f"Error starting Llamafile server: {str(e)}")
            if self.process:
                logger.info("Terminating process due to startup error")
                self.process.terminate()
                self._is_running = False
            raise LlamafileServerError(
                status_code=4, message=f"Failed to start server: {str(e)}"
            ) from e

    def _wait_for_ready(self, timeout: int) -> None:
        """Wait for the server to be ready.

        :param timeout: Timeout in seconds
        :raises LlamafileServerError: If server doesn't start within timeout
        """
        start_time = time.time()
        while time.time() - start_time < timeout:
            if self.is_server_running():
                logger.info("Llamafile server started successfully")
                return

            # Check if the process has terminated
            if self.process and self.process.poll() is not None:
                stdout, stderr = self.process.communicate()
                logger.error(
                    f"Process terminated unexpectedly. "
                    f"Exit code: {self.process.returncode}"
                )
                logger.error(f"Stdout: {stdout}")
                logger.error(f"Stderr: {stderr}")
                self._is_running = False
                raise LlamafileServerError(
                    status_code=5,
                    message=f"Process terminated unexpectedly. "
                    f"Exit code: {self.process.returncode}"
                )

            time.sleep(1)
            logger.debug(f"Waiting for server... (Elapsed: {time.time() - start_time:.2f}s)")

        # Server didn't start in time
        logger.error(f"Server failed to start within {timeout} seconds")
        self.stop()
        raise LlamafileServerError(
            status_code=6, message=f"Server failed to start within {timeout} seconds"
        )

    def stop(self) -> None:
        """Stop the Llamafile server."""
        if not self._is_running:
            logger.warning("Server is not running")
            return

        logger.info("Stopping Llamafile server...")
        if self.process:
            cleanup_llamafile(weakref.ref(self.process))
            self.process = None

        self._is_running = False
        logger.info("Llamafile server stopped")

    def restart(self, wait_for_ready: bool = True, timeout: int = 60) -> None:
        """Restart the Llamafile server.

        :param wait_for_ready: Whether to wait for the server to be ready
        :param timeout: Timeout in seconds to wait for server startup
        """
        logger.info("Restarting Llamafile server...")
        self.stop()
        time.sleep(2)  # Give it a moment to clean up
        self.start(wait_for_ready, timeout)

    def get_status(self) -> dict[str, Any]:
        """Get the current status of the server.

        :return: Dictionary containing server status information
        """
        return {
            "is_running": self._is_running,
            "is_responding": self.is_server_running() if self._is_running else False,
            "host": self.host,
            "port": self.port,
            "base_url": f"http://{self.host}:{self.port}/v1",
            "pid": self.process.pid if self.process else None,
        }

    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()

    def __del__(self):
        """Destructor to ensure cleanup."""
        try:
            self.stop()
        except Exception as e:
            logger.warning(f"Error during cleanup in __del__: {str(e)}")


def main():
    """Main function for running the standalone Llamafile server."""
    parser = argparse.ArgumentParser(description="Standalone Llamafile Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8085, help="Port to bind to")
    parser.add_argument("--cache-dir", default="~/.cache/flow-judge", help="Cache directory")
    parser.add_argument("--gpu-layers", type=int, default=34, help="Number of GPU layers")
    parser.add_argument("--context-size", type=int, default=8192, help="Context size")
    parser.add_argument("--threads", type=int, help="Number of threads")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--max-requests", type=int, default=1, help="Max concurrent requests")
    parser.add_argument("--temperature", type=float, default=0.1, help="Sampling temperature")
    parser.add_argument("--max-tokens", type=int, default=1000, help="Max new tokens")
    parser.add_argument("--disable-kv-offload", action="store_true", help="Disable KV offload")
    parser.add_argument("--no-quantized-kv", action="store_true", help="Disable quantized KV")
    parser.add_argument("--no-flash-attn", action="store_true", help="Disable flash attention")
    parser.add_argument("--log-level", default="INFO", help="Logging level")

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Create and start server
    server = LlamafileServer(
        cache_dir=os.path.expanduser(args.cache_dir),
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
        logger.info(f"Server running at http://{args.host}:{args.port}")
        logger.info("Press Ctrl+C to stop the server")

        # Keep the server running
        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        logger.info("Received interrupt signal, stopping server...")
        server.stop()
    except Exception as e:
        logger.error(f"Server error: {e}")
        server.stop()
        exit(1)


if __name__ == "__main__":
    main()
