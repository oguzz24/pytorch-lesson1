import torch


def task_1_1():
    """Create the required tensors."""
    random_tensor = torch.rand(3, 4)
    zeros_tensor = torch.zeros(2, 3, 4)
    ones_tensor = torch.ones(5, 5)
    reshape_tensor = torch.arange(16).reshape(4, 4)

    print("\n3x4 random tensor:")
    print(random_tensor)

    print("\n2x3x4 zeros tensor:")
    print(zeros_tensor)

    print("\n5x5 ones tensor:")
    print(ones_tensor)

    print("\n4x4 tensor from 0 to 15:")
    print(reshape_tensor)


def task_1_2():
    """Perform the required tensor operations."""
    A = torch.rand(3, 4)
    B = torch.rand(4, 3)

    if A.shape != (3, 4):
        raise ValueError("A must have shape 3x4.")

    if B.shape != (4, 3):
        raise ValueError("B must have shape 4x3.")

    print("\nTensor A:")
    print(A)

    print("\nTensor B:")
    print(B)

    print("\nTransposed A:")
    print(A.T)

    print("\nMatrix multiplication A x B:")
    print(torch.matmul(A, B))

    print("\nElement-wise multiplication A * B.T:")
    print(A * B.T)

    print("\nSum of all elements of A:")
    print(A.sum())


def task_1_3():
    """Extract values using indexing and slicing."""
    tensor_3d = torch.arange(125).reshape(5, 5, 5)

    if tensor_3d.shape != (5, 5, 5):
        raise ValueError("Tensor must have shape 5x5x5.")

    print("\n5x5x5 tensor:")
    print(tensor_3d)

    print("\nFirst row:")
    print(tensor_3d[0, 0, :])

    print("\nLast column:")
    print(tensor_3d[:, :, -1])

    print("\n2x2 submatrix from the center:")
    print(tensor_3d[2, 1:3, 1:3])

    print("\nElements with even indices:")
    print(tensor_3d[::2, ::2, ::2])


def task_1_4():
    """Reshape a tensor containing 24 elements."""
    tensor_24 = torch.arange(24)

    if tensor_24.numel() != 24:
        raise ValueError("Tensor must contain 24 elements.")

    print("\nShape 2x12:")
    print(tensor_24.reshape(2, 12))

    print("\nShape 3x8:")
    print(tensor_24.reshape(3, 8))

    print("\nShape 4x6:")
    print(tensor_24.reshape(4, 6))

    print("\nShape 2x3x4:")
    print(tensor_24.reshape(2, 3, 4))

    print("\nShape 2x2x2x3:")
    print(tensor_24.reshape(2, 2, 2, 3))


def run_tests():
    """Run simple tests for the required tensor operations."""
    A = torch.rand(3, 4)
    B = torch.rand(4, 3)
    tensor_24 = torch.arange(24)

    assert torch.rand(3, 4).shape == (3, 4)
    assert torch.zeros(2, 3, 4).shape == (2, 3, 4)
    assert torch.ones(5, 5).shape == (5, 5)
    assert torch.arange(16).reshape(4, 4).shape == (4, 4)

    assert A.T.shape == (4, 3)
    assert torch.matmul(A, B).shape == (3, 3)
    assert (A * B.T).shape == (3, 4)

    assert tensor_24.reshape(2, 12).shape == (2, 12)
    assert tensor_24.reshape(3, 8).shape == (3, 8)
    assert tensor_24.reshape(4, 6).shape == (4, 6)
    assert tensor_24.reshape(2, 3, 4).shape == (2, 3, 4)
    assert tensor_24.reshape(2, 2, 2, 3).shape == (2, 2, 2, 3)

    print("\nAll tests passed successfully.")


def main():
    """Run all tensor homework tasks."""
    task_1_1()
    task_1_2()
    task_1_3()
    task_1_4()
    run_tests()


if __name__ == "__main__":
    main()