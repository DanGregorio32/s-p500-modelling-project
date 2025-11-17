import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Cubic Spline 
def natural_cubic_spline(x, y, xs):
    n = len(x) - 1
    h = np.diff(x)
    alpha = [0] + [3*(y[i+1]-y[i])/h[i] - 3*(y[i]-y[i-1])/h[i-1] for i in range(1, n)] + [0]
    l = np.ones(n+1)
    mu = np.zeros(n+1)
    z = np.zeros(n+1)
    for i in range(1, n):
        l[i] = 2*(x[i+1]-x[i-1]) - h[i-1]*mu[i-1]
        mu[i] = h[i]/l[i]
        z[i] = (alpha[i] - h[i-1]*z[i-1])/l[i]
    b = np.zeros(n)
    c = np.zeros(n+1)
    d = np.zeros(n)
    for j in range(n-1, -1, -1):
        c[j] = z[j] - mu[j]*c[j+1]
        b[j] = (y[j+1]-y[j])/h[j] - h[j]*(c[j+1]+2*c[j])/3
        d[j] = (c[j+1]-c[j])/(3*h[j])

    # Store spline coefficients for each interval
    coeffs = []
    for j in range(n):
        coeffs.append((y[j], b[j], c[j], d[j], x[j]))
    # Evaluate spline at xs
    ys = []
    for xi in xs:
        for j in range(n):
            if x[j] <= xi <= x[j+1]:
                dx = xi - x[j]
                val = y[j] + b[j]*dx + c[j]*dx**2 + d[j]*dx**3
                ys.append(val)
                break
    return np.array(ys), coeffs

# Spline Derivative Evaluation 
def spline_first_derivative(coeffs, xi):
    # coeffs: list of (a, b, c, d, xj)
    for a, b, c, d, xj in coeffs:
        if xj <= xi <= xj+1:
            dx = xi - xj
            return b + 2*c*dx + 3*d*dx**2
    # If xi is at the end
    a, b, c, d, xj = coeffs[-1]
    dx = xi - xj
    return b + 2*c*dx + 3*d*dx**2

def spline_second_derivative(coeffs, xi):
    for a, b, c, d, xj in coeffs:
        if xj <= xi <= xj+1:
            dx = xi - xj
            return 2*c + 6*d*dx
    a, b, c, d, xj = coeffs[-1]
    dx = xi - xj
    return 2*c + 6*d*dx

# Newton-Raphson Method 
def newton_raphson(f, fprime, x0, tol=1e-5, maxiter=20):
    x = x0
    for _ in range(maxiter):
        fx = f(x)
        fpx = fprime(x)
        if abs(fpx) < 1e-10:
            break
        x_new = x - fx / fpx
        if abs(x_new - x) < tol:
            return x_new
        x = x_new
    return x

# Load Data
df = pd.read_csv('sp500_2020_2025_monthly.csv')
df['Date'] = pd.to_datetime(df['Date'], utc=True).dt.tz_localize(None)
df_2024 = df[(df['Date'] >= '2024-01-01') & (df['Date'] <= '2024-12-31')]
months = df_2024['Date'].dt.strftime('%b %Y').tolist()
y = df_2024['Close'].values
x = np.arange(len(months))

# Fit Spline 
x_fine = np.linspace(x.min(), x.max(), 500)
y_spline, spline_coeffs = natural_cubic_spline(x, y, x_fine)

#  Numerical Derivative
dy = np.array([spline_first_derivative(spline_coeffs, xi) for xi in x_fine])

# Find sign changes
roots = []
for i in range(len(x_fine)-1):
    if dy[i]*dy[i+1] < 0:
        # Root bracketed between x_fine[i] and x_fine[i+1]
        x0 = (x_fine[i] + x_fine[i+1]) / 2
        # Newton-Raphson refinement
        f = lambda z: spline_first_derivative(spline_coeffs, z)
        fprime = lambda z: spline_second_derivative(spline_coeffs, z)
        root = newton_raphson(f, fprime, x0)
        # Check root is within the interval
        if x.min() <= root <= x.max():
            roots.append(root)

# Identify maxima 
maxima = []
for r in roots:
    if spline_second_derivative(spline_coeffs, r) < 0:
        maxima.append((r, np.interp(r, x_fine, y_spline)))

# Plot
plt.plot(x, y, 'o', label='Actual Data')
plt.plot(x_fine, y_spline, '-', label='Cubic Spline')
for r, val in maxima:
    plt.plot(r, val, 'rx', markersize=10, label='Maxima')
plt.title('S&P 500 Closing Price (2024) and Maxima')
plt.xlabel('Month')
plt.ylabel('Closing Price')
plt.xticks(ticks=x, labels=months, rotation=45)
plt.legend()
plt.tight_layout()
plt.show()

# Print maxima
for r, val in maxima:
    print(f"Maxima at x={r:.2f} (approx {months[int(round(r))]}) with closing price {val:.2f}")
