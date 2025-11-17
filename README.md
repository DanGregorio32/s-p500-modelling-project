# s&p500-modelling-project
S&P 500 Index Modelling and Growth Analysis Using Python

This project analyses monthly S&P 500 closing prices from 2020 to 2024 using Python.

It was completed as part of my first year Finance and Data Analytics coursework (computational methods).

The analysis focuses on:
1. Modelling index behaviour
Polynomial approximation and cubic spline interpolation were used to model the index for two windows
February to May 2021
February to April 2023
2. Identifying local maxima in 2024
A spline model was fitted to the 2024 data.
Numerical differentiation and Newton-Raphson were used to locate critical points
Second derivative checks classified turning points.
3. Estimating cumulative growth
Spline interpolation combined with Simpson’s rule was used to approximate cumulative growth in 2023 and 2024
Results were compared across years
Skills demonstrated
Python, pandas, numpy, matplotlib
Spline interpolation, polynomial modelling
Numerical differentiation, Newton-Raphson
Numerical integration, time series analysis

