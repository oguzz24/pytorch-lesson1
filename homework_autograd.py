import torch


def task_2_1():
    """Calculate gradients of a multivariable function."""
    x = torch.tensor(1.0, requires_grad=True)
    y = torch.tensor(2.0, requires_grad=True)
    z = torch.tensor(3.0, requires_grad=True)

    f = x**2 + y**2 + z**2 + 2 * x * y * z
    f.backward()

    dx = 2 * x.detach() + 2 * y.detach() * z.detach()
    dy = 2 * y.detach() + 2 * x.detach() * z.detach()
    dz = 2 * z.detach() + 2 * x.detach() * y.detach()

    print("\nTask 2.1")
    print("f =", f.item())
    print("df/dx =", x.grad.item())
    print("df/dy =", y.grad.item())
    print("df/dz =", z.grad.item())

    assert torch.isclose(x.grad, dx)
    assert torch.isclose(y.grad, dy)
    assert torch.isclose(z.grad, dz)


def mse_loss(y_pred, y_true):
    """Calculate mean squared error."""
    if y_pred.shape != y_true.shape:
        raise ValueError("Tensor shapes must be equal.")

    if not y_pred.is_floating_point() or not y_true.is_floating_point():
        raise TypeError("Tensors must contain floating point values.")

    return torch.mean((y_pred - y_true) ** 2)


def task_2_2():
    """Calculate gradients of MSE with respect to w and b."""
    x = torch.tensor([1.0, 2.0, 3.0])
    y_true = torch.tensor([2.0, 4.0, 6.0])

    w = torch.tensor(1.0, requires_grad=True)
    b = torch.tensor(0.0, requires_grad=True)

    y_pred = w * x + b
    loss = mse_loss(y_pred, y_true)

    loss.backward()

    print("\nTask 2.2")
    print("MSE =", loss.item())
    print("dMSE/dw =", w.grad.item())
    print("dMSE/db =", b.grad.item())

    expected_w_grad = torch.tensor(-28.0 / 3.0)
    expected_b_grad = torch.tensor(-4.0)

    assert torch.isclose(w.grad, expected_w_grad)
    assert torch.isclose(b.grad, expected_b_grad)


def task_2_3():
    """Calculate a gradient using the chain rule."""
    x = torch.tensor(2.0, requires_grad=True)

    f = torch.sin(x**2 + 1)
    f.backward()

    backward_gradient = x.grad.clone()
    analytical_gradient = 2 * x.detach() * torch.cos(x.detach() ** 2 + 1)

    x_check = torch.tensor(2.0, requires_grad=True)
    f_check = torch.sin(x_check**2 + 1)
    autograd_gradient = torch.autograd.grad(f_check, x_check)[0]

    print("\nTask 2.3")
    print("f =", f.item())
    print("df/dx =", backward_gradient.item())
    print("Analytical gradient =", analytical_gradient.item())
    print("torch.autograd.grad =", autograd_gradient.item())

    assert torch.isclose(backward_gradient, analytical_gradient)
    assert torch.isclose(backward_gradient, autograd_gradient)


def main():
    task_2_1()
    task_2_2()
    task_2_3()
    print("\nAll tests passed successfully.")


if __name__ == "__main__":
    main()