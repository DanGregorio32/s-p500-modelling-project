import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from scipy.optimize import newton


# LOAD AND CLEAN DATA
df = pd.read_csv('sp500_2020_2025_monthly.csv')
df['Date'] = pd.to_datetime(df['Date'], utc=True).dt.tz_localize(None)

# FILTER DATA FOR THE TWO PERIODS
period1 = df[(df['Date'] >= '2021-02-01') & (df['Date'] <= '2021-05-31')][['Date', 'Close']]
period2 = df[(df['Date'] >= '2023-02-01') & (df['Date'] <= '2023-04-30')][['Date', 'Close']]

# PREPARE DATA FOR MODELING
months1 = period1['Date'].dt.strftime('%b %Y').tolist()
y1 = period1['Close'].values
months2 = period2['Date'].dt.strftime('%b %Y').tolist()
y2 = period2['Close'].values

x1 = np.arange(len(months1))
x2 = np.arange(len(months2))
x1_smooth = np.linspace(x1.min(), x1.max(), 100)
x2_smooth = np.linspace(x2.min(), x2.max(), 100)

# CUBIC SPLINE INTERPOLATION
cubic_spline_model1 = CubicSpline(x1, y1)
y1_cubic_spline = cubic_spline_model1(x1_smooth)
cubic_spline_model2 = CubicSpline(x2, y2)
y2_cubic_spline = cubic_spline_model2(x2_smooth)

# POLYNOMIAL APPROXIMATION
poly1 = np.poly1d(np.polyfit(x1, y1, 3))
y1_poly = poly1(x1_smooth)
poly2 = np.poly1d(np.polyfit(x2, y2, 3))
y2_poly = poly2(x2_smooth)

# PLOT RESULTS
plt.figure(figsize=(14, 6))

# Plot for Period 1 (Feb-May 2021)
plt.subplot(1, 2, 1)
plt.plot(x1, y1, 'o', label='Actual Data')
plt.plot(x1_smooth, y1_cubic_spline, '-', label='Cubic Spline')
plt.plot(x1_smooth, y1_poly, '-.', label='Polynomial Approximation')
plt.title('Closing Price: Feb-May 2021')
plt.xlabel('Month')
plt.ylabel('Closing Price')
plt.xticks(ticks=x1, labels=months1)
plt.legend()

# Plot for Period 2 (Feb-Apr 2023)
plt.subplot(1, 2, 2)
plt.plot(x2, y2, 'o', label='Actual Data')
plt.plot(x2_smooth, y2_cubic_spline, '-', label='Cubic Spline')
plt.plot(x2_smooth, y2_poly, '-.', label='Polynomial Approximation')
plt.title('Closing Price: Feb-Apr 2023')
plt.xlabel('Month')
plt.ylabel('Closing Price')
plt.xticks(ticks=x2, labels=months2)
plt.legend()

plt.tight_layout()
plt.show()







import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from scipy.optimize import newton

# Load and filter 2024 data
df = pd.read_csv('sp500_2020_2025_monthly.csv')
df['Date'] = pd.to_datetime(df['Date'], utc=True).dt.tz_localize(None)
df_2024 = df[(df['Date'] >= '2024-01-01') & (df['Date'] <= '2024-12-31')]

months = df_2024['Date'].dt.strftime('%b %Y').tolist()
y = df_2024['Close'].values
x = np.arange(len(months))

# Fit cubic spline
spline = CubicSpline(x, y)

# Derivative of the spline
spline_deriv = spline.derivative()

# Find critical points (roots of the derivative)
# Use midpoints as initial guesses for Newton-Raphson
roots = []
for i in range(len(x)-1):
    try:
        root = newton(spline_deriv, x0=(x[i]+x[i+1])/2)
        # Check if root is within the interval
        if x[i] < root < x[i+1]:
            roots.append(root)
    except RuntimeError:
        continue

# Evaluate maxima/minima
maxima = []
for r in roots:
    second_deriv = spline.derivative(2)(r)
    if second_deriv < 0:  # Maximum
        maxima.append((r, spline(r)))

# Plot
x_smooth = np.linspace(x.min(), x.max(), 200)
plt.plot(x, y, 'o', label='Actual Data')
plt.plot(x_smooth, spline(x_smooth), '-', label='Cubic Spline')
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

print(df)
