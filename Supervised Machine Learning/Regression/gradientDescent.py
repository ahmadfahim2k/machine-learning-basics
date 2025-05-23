import numpy as np
import pandas as pd

def gradient_descent(x, y, lr=0.1, epochs=1000):
    m,b = 0.0,0.0
    
    x_min, y_min = x.min(), y.min()
    x_max, y_max = x.max(), y.max()
    
    x_scaled = (x - x_min) / (x_max - x_min)
    y_scaled = (y - y_min) / (y_max - y_min)
    
    for epoch in range(epochs):
        y_pred = m*x_scaled+b
        error = y_scaled - y_pred
        cost = np.mean(error ** 2)
        
        dm = -2 * np.mean(error*x_scaled)
        db = -2 * np.mean(error)
        
        b -= db * lr
        m -= dm * lr
        
        if(epoch % 100 == 0):
            print(f"Epoch {epoch}: Cost = {cost}, m = {m}, b = {b}")
        
    # Scale back the coefficients to original scale
    b_original = b * (y_max - y_min) + y_min - m * (y_max - y_min) * x_min / (x_max - x_min)
    m_original = m * (y_max - y_min) / (x_max - x_min)

    return b_original, m_original

if __name__ == '__main__':
    df = pd.read_csv("D:/Projects/Python/Machine Learning Basics/machine-learning-basics/Supervised Machine Learning/home_prices.csv")

    x = df['area_sqr_ft'].to_numpy()
    y = df['price_lakhs'].to_numpy()
    
    b, m = gradient_descent(x, y)

    print(f"Final Results: m={m}, b={b}")