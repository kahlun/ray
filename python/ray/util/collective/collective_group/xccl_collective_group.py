"""XCCL (Intel GPU) collective group implementation for Ray.

This module provides XCCL collective communication support for Intel GPU (XPU) devices.
It follows the same architecture as NCCLGroup but uses PyTorch's ProcessGroupXCCL backend
for device-to-device communication via Intel's oneCCL library.
"""

import datetime
import logging
import os
from typing import Callable, List, Optional

import torch
import torch.distributed as dist

from ray.util.collective.collective_group.base_collective_group import BaseGroup
from ray.util.collective.const import get_store_name
from ray.util.collective.types import (
    AllGatherOptions,
    AllReduceOptions,
    Backend,
    BarrierOptions,
    BroadcastOptions,
    RecvOptions,
    ReduceOptions,
    ReduceScatterOptions,
    SendOptions,
)

logger = logging.getLogger(__name__)

try:
    import torch

    # Check if torch has XPU support
    if hasattr(torch, "xpu"):
        _XCCL_AVAILABLE = True
        _LOG_XCCL_WARNING = False
    else:
        _XCCL_AVAILABLE = False
        _LOG_XCCL_WARNING = True
except ImportError:
    _XCCL_AVAILABLE = False
    _LOG_XCCL_WARNING = True


class _XcclProcessGroup:
    """Standalone XCCL process group wrapper for distributed training on Intel GPUs.

    This wraps PyTorch's ProcessGroupXCCL backend to provide collective operations
    across XPU devices. Each instance manages a single communication group with its own
    TCPStore for rank synchronization.

    Args:
        master_address: The master node's IP address for TCPStore
        master_port: The TCP port for the store
        rank: The rank of this process in the group
        world_size: Total number of processes in the group
        device: The XPU device to use (int or torch.device)
    """

    def __init__(
        self,
        master_address: str,
        master_port: int,
        rank: int,
        world_size: int,
        device,
    ):
        from torch._C._distributed_c10d import PrefixStore, ProcessGroup
        from torch.distributed.distributed_c10d import ProcessGroupXCCL

        self.rank = rank
        self.world_size = world_size
        # Normalize device to torch.device format
        self.device = (
            device if isinstance(device, torch.device) else torch.device("xpu", int(device))
        )

        timeout = datetime.timedelta(
            seconds=int(os.getenv("VERL_XCCL_TIMEOUT_S", "1800"))
        )

        # Rank 0 owns the TCPStore server; others connect as clients
        store = dist.TCPStore(
            host_name=master_address,
            port=master_port,
            world_size=world_size,
            is_master=(rank == 0),
            timeout=timeout,
        )

        # Prefix isolates this group's store keys from other torch.distributed traffic
        prefix_store = PrefixStore("ray_xccl_collective", store)

        # Create the process group and register the XCCL backend
        pg = ProcessGroup(prefix_store, rank, world_size)
        opts = ProcessGroupXCCL.Options()
        opts._timeout = timeout
        backend_class = ProcessGroupXCCL(prefix_store, rank, world_size, opts)
        backend_type = ProcessGroup.BackendType.XCCL
        pg._set_default_backend(backend_type)
        backend_class._set_sequence_number_for_group()
        pg._register_backend(self.device, backend_type, backend_class)

        self.pg = pg
        self.comm = pg

    def broadcast(self, tensor: torch.Tensor, src: int = 0):
        """Broadcast a tensor from source rank to all others."""
        opts = dist.BroadcastOptions()
        opts.rootRank = src
        self.pg.broadcast([tensor], opts).wait()

    def allreduce(self, tensor: torch.Tensor, op=dist.ReduceOp.SUM):
        """All-reduce a tensor across all ranks."""
        opts = dist.AllreduceOptions()
        opts.reduceOp = op
        self.pg.allreduce([tensor], opts).wait()

    def allgather(self, output_tensors: List[torch.Tensor], input_tensor: torch.Tensor):
        """All-gather tensors from all ranks."""
        opts = dist.AllGatherOptions()
        self.pg.allgather(output_tensors, [input_tensor], opts).wait()

    def reducescatter(
        self, output_tensor: torch.Tensor, input_tensors: List[torch.Tensor]
    ):
        """Reduce-scatter tensors across all ranks."""
        opts = dist.ReduceScatterOptions()
        self.pg.reducescatter([output_tensor], input_tensors, opts).wait()

    def reduce(self, tensor: torch.Tensor, dst: int, op=dist.ReduceOp.SUM):
        """Reduce a tensor to destination rank."""
        opts = dist.ReduceOptions()
        opts.rootRank = dst
        opts.reduceOp = op
        self.pg.reduce([tensor], opts).wait()

    def send(self, tensor: torch.Tensor, dst: int):
        """Send a tensor to destination rank (point-to-point)."""
        opts = dist.SendOptions()
        opts.destRank = dst
        self.pg.send([tensor], opts).wait()

    def recv(self, tensor: torch.Tensor, src: int):
        """Receive a tensor from source rank (point-to-point)."""
        opts = dist.RecvOptions()
        opts.srcRank = src
        self.pg.recv([tensor], opts).wait()

    def destroyComm(self, comm=None):
        """Destroy/cleanup the process group."""
        try:
            self.pg.shutdown()
        except Exception as e:
            logger.warning(f"XCCL destroy_process_group failed: {e}")


class XCCLGroup(BaseGroup):
    """Intel GPU (XPU) collective group using XCCL backend.

    This class provides collective communication operations (allreduce, broadcast, etc.)
    for Intel GPU workloads. It manages a single process group per instance and delegates
    collective operations to the ProcessGroupXCCL backend.

    This implementation is simpler than NCCLGroup because:
    - XPU uses a single device-context (unlike CUDA which can have multiple GPUs)
    - No need for stream/event management (ProcessGroupXCCL handles kernel scheduling)
    - No need for cupy buffer management (torch.Tensor is sufficient)
    """

    def __init__(self, world_size: int, rank: int, group_name: str):
        """Initialize an XCCL collective group.

        Args:
            world_size: Total number of processes
            rank: Rank of current process
            group_name: Name of the collective group
        """
        super().__init__(world_size, rank, group_name)
        self._process_group = None
        self.master_address = None
        self.master_port = None

    def destroy_group(self):
        """Destroy the group and release XCCL resources."""
        if self._process_group is not None:
            self._process_group.destroyComm()
            self._process_group = None
        super().destroy_group()

    @classmethod
    def backend(cls):
        """Return the backend type."""
        return Backend.XCCL

    @classmethod
    def check_backend_availability(cls) -> bool:
        """Check if XCCL backend is available."""
        global _LOG_XCCL_WARNING, _XCCL_AVAILABLE
        if _LOG_XCCL_WARNING:
            logger.warning(
                "XCCL (Intel GPU) is not available. Please install PyTorch "
                "with XPU support following Intel's DPK installation guide: "
                "https://github.com/intel/intel-extension-for-pytorch"
            )
            _LOG_XCCL_WARNING = False
        return _XCCL_AVAILABLE

    def _ensure_initialized(self):
        """Ensure the process group is initialized."""
        if self._process_group is None:
            raise RuntimeError(
                "XCCL process group not initialized. "
                "Call init_communicators() first."
            )

    def init_communicators(
        self, master_address: str, master_port: int, device: Optional[int] = None
    ):
        """Initialize the XCCL communicators.

        Args:
            master_address: Master node IP address
            master_port: Master node TCP port
            device: XPU device index (uses current device if not specified)
        """
        if device is None:
            device = torch.xpu.current_device()

        self._process_group = _XcclProcessGroup(
            master_address, master_port, self._rank, self._world_size, device
        )
        self.master_address = master_address
        self.master_port = master_port
        logger.debug(
            f"XCCL communicators initialized: rank={self._rank}, "
            f"world_size={self._world_size}, device={device}"
        )

    def allreduce(
        self,
        tensors: list,
        allreduce_options: AllReduceOptions = AllReduceOptions(),
    ):
        """Allreduce tensors across the group.

        Args:
            tensors: List of tensors to reduce (one per device)
            allreduce_options: Allreduce options (reduce operation, etc.)
        """
        self._ensure_initialized()
        for tensor in tensors:
            self._process_group.allreduce(
                tensor, op=allreduce_options.reduceOp
            )

    def barrier(self, barrier_options: BarrierOptions = BarrierOptions()):
        """Synchronize all processes in the group."""
        self._ensure_initialized()
        # Use allreduce with a dummy tensor as barrier
        barrier_tensor = torch.tensor([1], dtype=torch.int64, device="xpu")
        self._process_group.allreduce(barrier_tensor)

    def reduce(self, tensors: list, reduce_options: ReduceOptions = ReduceOptions()):
        """Reduce tensors to destination rank.

        Args:
            tensors: List of tensors to reduce
            reduce_options: Reduce options (destination rank, reduce op, etc.)
        """
        self._ensure_initialized()
        for tensor in tensors:
            self._process_group.reduce(
                tensor, dst=reduce_options.root_rank, op=reduce_options.reduceOp
            )

    def broadcast(
        self,
        tensors: list,
        broadcast_options: BroadcastOptions = BroadcastOptions(),
    ):
        """Broadcast tensors from source rank to all others.

        Args:
            tensors: List of tensors to broadcast
            broadcast_options: Broadcast options (source rank, etc.)
        """
        self._ensure_initialized()
        for tensor in tensors:
            self._process_group.broadcast(tensor, src=broadcast_options.root_rank)

    def allgather(
        self,
        tensor_lists: list,
        tensors: list,
        allgather_options: AllGatherOptions = AllGatherOptions(),
    ):
        """Allgather tensors across all ranks.

        Args:
            tensor_lists: Output tensor lists (one list per rank)
            tensors: Input tensors to gather
            allgather_options: Allgather options
        """
        self._ensure_initialized()
        for tensor_list, tensor in zip(tensor_lists, tensors):
            self._process_group.allgather(tensor_list, tensor)

    def reducescatter(
        self,
        tensors: list,
        tensor_lists: list,
        reducescatter_options: ReduceScatterOptions = ReduceScatterOptions(),
    ):
        """Reduce-scatter tensors across all ranks.

        Args:
            tensors: Output tensors (scattered from each rank's result)
            tensor_lists: Input tensor lists to scatter
            reducescatter_options: Reduce-scatter options
        """
        self._ensure_initialized()
        for tensor, tensor_list in zip(tensors, tensor_lists):
            self._process_group.reducescatter(tensor, tensor_list)

    def send(self, tensor: list, send_options: SendOptions):
        """Send a tensor to destination rank (point-to-point).

        Args:
            tensor: List with single tensor to send
            send_options: Send options (destination rank, etc.)
        """
        self._ensure_initialized()
        if len(tensor) != 1:
            raise ValueError("Send expects a list with a single tensor")
        self._process_group.send(tensor[0], dst=send_options.dest_rank)

    def recv(self, tensor: list, recv_options: RecvOptions):
        """Receive a tensor from source rank (point-to-point).

        Args:
            tensor: List with single tensor buffer to receive into
            recv_options: Recv options (source rank, etc.)
        """
        self._ensure_initialized()
        if len(tensor) != 1:
            raise ValueError("Recv expects a list with a single tensor")
        self._process_group.recv(tensor[0], src=recv_options.src_rank)
