import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv('sp500_2020_2025_monthly.csv')
df['Date'] = pd.to_datetime((df['Date']), utc=True).dt.tz_localize(None)

# Filter for 2023 and 2024
df_2023 = df[(df['Date'] >= '2023-01-01') & (df['Date'] <= '2023-12-31')]
df_2024 = df[(df['Date'] >= '2024-01-01') & (df['Date'] <= '2024-12-31')]

def polyfit(x, y, degree):
    X = np.vander(x, degree+1, increasing=False)
    coeffs = np.linalg.lstsq(X, y, rcond=None)[0]
    return coeffs

def polyval(coeffs, x):
    return np.polyval(coeffs, x)

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
    ys = []
    for xi in xs:
        for j in range(n):
            if x[j] <= xi <= x[j+1]:
                dx = xi - x[j]
                val = y[j] + b[j]*dx + c[j]*dx**2 + d[j]*dx**3
                ys.append(val)
                break
    return np.array(ys)

# Prepare and fit for 2023
months_2023 = df_2023['Date'].dt.strftime('%b %Y').tolist()
y_2023 = df_2023['Close'].values
x_2023 = np.arange(len(months_2023))
x_2023_fine = np.linspace(x_2023.min(), x_2023.max(), 200)
coeffs_2023 = polyfit(x_2023, y_2023, 3)
y_2023_poly = polyval(coeffs_2023, x_2023_fine)
y_2023_spline = natural_cubic_spline(x_2023, y_2023, x_2023_fine)

# Prepare and fit for 2024
months_2024 = df_2024['Date'].dt.strftime('%b %Y').tolist()
y_2024 = df_2024['Close'].values
x_2024 = np.arange(len(months_2024))
x_2024_fine = np.linspace(x_2024.min(), x_2024.max(), 200)
coeffs_2024 = polyfit(x_2024, y_2024, 3)
y_2024_poly = polyval(coeffs_2024, x_2024_fine)
y_2024_spline = natural_cubic_spline(x_2024, y_2024, x_2024_fine)

def simpsons_rule(y, x):
    n = len(x) - 1
    if n % 2 == 1:
        n -= 1  
    h = (x[n] - x[0]) / n
    result = y[0] + y[n]
    for i in range(1, n, 2):
        result += 4 * y[i]
    for i in range(2, n-1, 2):
        result += 2 * y[i]
    return result * h / 3

# Numerically integrate the spline curve for each year
cum_growth_2023 = simpsons_rule(y_2023_spline, x_2023_fine)
cum_growth_2024 = simpsons_rule(y_2024_spline, x_2024_fine)

print(f"Cumulative growth for 2023 (Simpson's rule): {cum_growth_2023:.2f}")
print(f"Cumulative growth for 2024 (Simpson's rule): {cum_growth_2024:.2f}")

# Plot
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(x_2023, y_2023, 'o', label='Actual Data 2023')
plt.plot(x_2023_fine, y_2023_poly, '--', label='Polyfit')
plt.plot(x_2023_fine, y_2023_spline, ':', label='Cubic Spline')
plt.fill_between(x_2023_fine, y_2023_spline, alpha=0.2)
plt.title('S&P 500 Closing Price 2023')
plt.xlabel('Month')
plt.ylabel('Closing Price')
plt.xticks(ticks=x_2023, labels=months_2023, rotation=45)
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(x_2024, y_2024, 'o', label='Actual Data 2024')
plt.plot(x_2024_fine, y_2024_poly, '--', label='Polyfit')
plt.plot(x_2024_fine, y_2024_spline, ':', label='Cubic Spline')
plt.fill_between(x_2024_fine, y_2024_spline, alpha=0.2)
plt.title('S&P 500 Closing Price 2024')
plt.xlabel('Month')
plt.ylabel('Closing Price')
plt.xticks(ticks=x_2024, labels=months_2024, rotation=45)
plt.legend()

plt.tight_layout()
plt.show()
