import torch
import torch.optim as optim
import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"

# ---- Metric ----
def metric(x):
    return 1 + torch.sum(x**2, dim=-1, keepdim=True)

# ---- Energy ----
def energy_curve(c):
    dc = c[1:] - c[:-1]
    speed2 = torch.sum(dc**2, dim=-1)
    mid = (c[1:] + c[:-1]) / 2
    g = metric(mid).squeeze()
    return torch.sum(g * speed2)

# ---- Piecewise linear ----
def optimize_piecewise(x0, x1, N=50, lr=1e-2, steps=2000):
    t = torch.linspace(0, 1, N, device=device).unsqueeze(1)
    c = x0*(1-t) + x1*t
    c = c.clone().detach().requires_grad_(True)

    optimizer = optim.Adam([c], lr=lr)

    for step in range(steps):
        optimizer.zero_grad()

        # fix endpoints
        c.data[0] = x0
        c.data[-1] = x1

        E = energy_curve(c)
        E.backward()
        optimizer.step()

        if step % 500 == 0:
            print(f"[Piecewise] Step {step}, Energy {E.item():.4f}")

    return c.detach()

# ---- Cubic parametrization ----
def cubic_curve(t, coeffs):
    return (
        coeffs[0]
        + coeffs[1]*t
        + coeffs[2]*t**2
        + coeffs[3]*t**3
    )

def optimize_cubic(x0, x1, N=100, lr=1e-2, steps=2000):
    t = torch.linspace(0, 1, N, device=device).unsqueeze(1)

    coeffs = torch.randn(4, 2, device=device, requires_grad=True)
    optimizer = optim.Adam([coeffs], lr=lr)

    for step in range(steps):
        optimizer.zero_grad()

        c = cubic_curve(t, coeffs)

        # boundary constraints
        c0 = cubic_curve(torch.tensor([[0.0]], device=device), coeffs)
        c1 = cubic_curve(torch.tensor([[1.0]], device=device), coeffs)
        loss_bc = torch.sum((c0 - x0)**2) + torch.sum((c1 - x1)**2)

        E = energy_curve(c)
        loss = E + 1000 * loss_bc

        loss.backward()
        optimizer.step()

        if step % 500 == 0:
            print(f"[Cubic] Step {step}, Energy {E.item():.4f}")

    return cubic_curve(t, coeffs).detach()

# ---- MAIN TEST ----
if __name__ == "__main__":
    torch.manual_seed(0)

    x0 = torch.tensor([0.0, 0.0], device=device)
    x1 = torch.tensor([1.5, 1.0], device=device)

    print("Optimizing piecewise...")
    curve_pw = optimize_piecewise(x0, x1)

    print("\nOptimizing cubic...")
    curve_cubic = optimize_cubic(x0, x1)

    # ---- Plot ----
    pw = curve_pw.cpu().numpy()
    cu = curve_cubic.cpu().numpy()

    plt.figure(figsize=(6,6))
    plt.plot(pw[:,0], pw[:,1], label="Piecewise")
    plt.plot(cu[:,0], cu[:,1], label="Cubic")
    plt.scatter([x0[0].item(), x1[0].item()],
                [x0[1].item(), x1[1].item()],
                c="red", label="Endpoints")

    plt.title("Geodesic Approximation (Energy Minimization)")
    plt.legend()
    plt.axis("equal")
    plt.grid()
    plt.savefig("geodesic_approximation.png")

