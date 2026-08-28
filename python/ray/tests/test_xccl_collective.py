"""Unit tests for XCCL collective communication backend.

These tests verify that Intel GPU (XPU) collective operations work correctly.
They follow the same pattern as NCCL tests but use torch.xpu devices instead of cuda.
"""

import pytest
import torch

import ray
import ray.util.collective
from ray.util.collective.types import Backend


# Check if XPU is available
def xpu_available():
    try:
        return hasattr(torch, "xpu") and torch.xpu.is_available()
    except (RuntimeError, AttributeError):
        return False


@pytest.fixture
def setup_ray():
    """Setup Ray for tests."""
    if not ray.is_initialized():
        ray.init(num_cpus=4)
    yield
    ray.shutdown()


@pytest.fixture
def skip_if_no_xpu():
    """Skip test if XPU is not available."""
    if not xpu_available():
        pytest.skip("XPU not available")


# Shape and dtype for tensors
SHAPE = (10, 10)
DTYPE = torch.float32


@ray.remote
class XPUActor:
    """Test actor with XPU support."""

    def __init__(self, shape, dtype):
        self.shape = shape
        self.dtype = dtype
        # Initialize on XPU device
        device = torch.device("xpu:0")
        self.tensor = torch.zeros(shape, dtype=dtype, device=device)

    def make_tensor(self, value=1.0):
        """Create a tensor filled with a value."""
        device = torch.device("xpu:0")
        self.tensor = torch.full(self.shape, value, dtype=self.dtype, device=device)

    def get_tensor(self):
        """Return the tensor (copied to CPU for comparison)."""
        return self.tensor.cpu()

    def init_group(self, world_size, rank, backend, group_name):
        """Initialize the collective group."""
        ray.util.collective.init_collective_group(
            world_size=world_size,
            rank=rank,
            backend=backend,
            group_name=group_name,
        )

    def allreduce_tensor(self, group_name):
        """Perform allreduce on the tensor."""
        ray.util.collective.allreduce(self.tensor, group_name=group_name)

    def broadcast_tensor(self, group_name):
        """Perform broadcast on the tensor."""
        ray.util.collective.broadcast(self.tensor, src_rank=0, group_name=group_name)

    def barrier(self, group_name):
        """Wait at barrier."""
        ray.util.collective.barrier(group_name=group_name)


@pytest.mark.skipif(not xpu_available(), reason="XPU not available")
def test_xccl_backend_available(setup_ray, skip_if_no_xpu):
    """Test that XCCL backend is registered."""
    assert ray.util.collective.is_backend_available("XCCL")


@pytest.mark.skipif(not xpu_available(), reason="XPU not available")
def test_xccl_allreduce(setup_ray, skip_if_no_xpu):
    """Test allreduce on XPU."""
    world_size = 2
    actors = [XPUActor.remote(SHAPE, DTYPE) for _ in range(world_size)]

    # Initialize each actor with its rank
    for rank, actor in enumerate(actors):
        ray.get(actor.init_group.remote(world_size, rank, "XCCL", "test_group"))

    # Set each tensor to rank value (0, 1, 2, ...)
    for rank, actor in enumerate(actors):
        ray.get(actor.make_tensor.remote(float(rank + 1)))

    # Perform allreduce (SUM)
    for actor in actors:
        ray.get(actor.allreduce_tensor.remote("test_group"))

    # Check results: sum should be 1 + 2 = 3 on all ranks
    expected_sum = sum(range(1, world_size + 1))
    tensors = ray.get([actor.get_tensor.remote() for actor in actors])

    for rank, tensor in enumerate(tensors):
        expected = torch.full(SHAPE, float(expected_sum), dtype=DTYPE)
        assert torch.allclose(
            tensor, expected, atol=1e-5
        ), f"Rank {rank} tensor does not match expected sum"


@pytest.mark.skipif(not xpu_available(), reason="XPU not available")
def test_xccl_broadcast(setup_ray, skip_if_no_xpu):
    """Test broadcast on XPU."""
    world_size = 2
    actors = [XPUActor.remote(SHAPE, DTYPE) for _ in range(world_size)]

    # Initialize each actor
    for rank, actor in enumerate(actors):
        ray.get(actor.init_group.remote(world_size, rank, "XCCL", "test_bcast"))

    # Set rank 0 to value 42, others to 0
    for rank, actor in enumerate(actors):
        value = 42.0 if rank == 0 else 0.0
        ray.get(actor.make_tensor.remote(value))

    # Perform broadcast from rank 0
    for actor in actors:
        ray.get(actor.broadcast_tensor.remote("test_bcast"))

    # Check all tensors are now 42
    tensors = ray.get([actor.get_tensor.remote() for actor in actors])
    expected = torch.full(SHAPE, 42.0, dtype=DTYPE)

    for rank, tensor in enumerate(tensors):
        assert torch.allclose(
            tensor, expected, atol=1e-5
        ), f"Rank {rank} tensor not broadcast correctly"


@pytest.mark.skipif(not xpu_available(), reason="XPU not available")
def test_xccl_barrier(setup_ray, skip_if_no_xpu):
    """Test barrier synchronization on XPU."""
    world_size = 2
    actors = [XPUActor.remote(SHAPE, DTYPE) for _ in range(world_size)]

    # Initialize each actor
    for rank, actor in enumerate(actors):
        ray.get(actor.init_group.remote(world_size, rank, "XCCL", "test_barrier"))

    # Call barrier (should not hang/deadlock)
    barrier_futures = [actor.barrier.remote("test_barrier") for actor in actors]
    ray.get(barrier_futures)  # All should complete


@pytest.mark.skipif(not xpu_available(), reason="XPU not available")
def test_xccl_multiple_groups(setup_ray, skip_if_no_xpu):
    """Test that multiple independent collective groups work."""
    world_size = 2

    # First group
    actors1 = [XPUActor.remote(SHAPE, DTYPE) for _ in range(world_size)]
    for rank, actor in enumerate(actors1):
        ray.get(actor.init_group.remote(world_size, rank, "XCCL", "group1"))

    # Second group
    actors2 = [XPUActor.remote(SHAPE, DTYPE) for _ in range(world_size)]
    for rank, actor in enumerate(actors2):
        ray.get(actor.init_group.remote(world_size, rank, "XCCL", "group2"))

    # Use first group
    for actor in actors1:
        ray.get(actor.make_tensor.remote(1.0))
        ray.get(actor.allreduce_tensor.remote("group1"))

    tensors1 = ray.get([actor.get_tensor.remote() for actor in actors1])

    # Use second group
    for actor in actors2:
        ray.get(actor.make_tensor.remote(2.0))
        ray.get(actor.allreduce_tensor.remote("group2"))

    tensors2 = ray.get([actor.get_tensor.remote() for actor in actors2])

    # Check group1: 1 + 1 = 2
    expected1 = torch.full(SHAPE, 2.0, dtype=DTYPE)
    for tensor in tensors1:
        assert torch.allclose(tensor, expected1, atol=1e-5)

    # Check group2: 2 + 2 = 4
    expected2 = torch.full(SHAPE, 4.0, dtype=DTYPE)
    for tensor in tensors2:
        assert torch.allclose(tensor, expected2, atol=1e-5)


@pytest.mark.skipif(not xpu_available(), reason="XPU not available")
def test_xccl_backend_type(setup_ray, skip_if_no_xpu):
    """Test that XCCL backend is correctly identified."""
    from ray.util.collective.collective_group.xccl_collective_group import XCCLGroup

    assert XCCLGroup.backend() == Backend.XCCL
    assert XCCLGroup.check_backend_availability() == xpu_available()


if __name__ == "__main__":
    # Quick manual test
    if xpu_available():
        print("XCCL tests can run on this system")
        print("Run with: pytest python/ray/tests/test_xccl_collective.py -v")
    else:
        print("XPU not available - skipping tests")
