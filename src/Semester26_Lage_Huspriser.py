import math
import pandas as pd
import numpy as np



def factor_distance(x1:int, y1:int, x2:int, y2:int) ->float:
    dist = np.sqrt((x1-x2)**2 + (y1-y2)**2)
    return math.exp(-dist*0.1)

def factor_pos_bergen(x:int, y:int)->float:
    w = (1
         + 1.5 * factor_distance(40, 58, x, y)  # Sentrum
         + 0.5 * factor_distance(40, 20, x, y)  # Fana
         )
    return w

def factor_pos_stavanger(x:int, y:int)->float:
    w = (1
         + 1.5 * factor_distance(75, 75, x, y)  # Sentrum
         + 0.5 * factor_distance(60, 30, x, y)  # Sandnes
         )
    return w

def factor_pos_oslo(x:int, y:int)->float:
    w = (1
         + 1.5 * factor_distance(45, 60, x, y)  # Sentrum
         + 0.5 * factor_distance(20, 60, x, y)  # Skøyen
         )
    return w

def factor_standard(standard:int)->float:
    return np.sqrt(standard)

def factor_small(m2):
    t = (m2 - 50) / (200 - 50)     # normalize 50→300 to 0→1
    t = np.clip(t, 0.5, 1)           # keep inside range
    return 1 - 0.5 * t             # linear inverse

def norm_price(m2:int)->float:
    factor = factor_small(m2)
    return m2 *  75.000 * factor

def factor_house_type(t:int)->float:
    if t==1:
        return 0.7
    if t==2:
        return 1.0
    if t==3:
        return 1.2
    return 1

def house_price_bergen(x:int, y:int, house_type:int, standard:int, m2:int)->float:
    price = (factor_pos_bergen(x, y)
             *factor_house_type(house_type)
             *factor_standard(standard)
             *norm_price(m2))
    return price

def house_price_stavanger(x:int, y:int, house_type:int, standard:int, m2:int)->float:
    price = (factor_pos_stavanger(x, y)
             *factor_house_type(house_type)
             *factor_standard(standard)
             *norm_price(m2))
    return price

def house_price_oslo(x:int, y:int, house_type:int, standard:int, m2:int)->float:
    price = (factor_pos_oslo(x, y)
             *factor_house_type(house_type)
             *factor_standard(standard)
             *norm_price(m2))
    return price

np.random.seed(1234)
samples = 256
xs = np.random.randint(0, 100, samples)
ys = np.random.randint(0, 100, samples)
house_types = np.round(np.random.uniform(1, 4, samples)).astype(int)
standards = np.random.uniform(1.0, 4.0, samples)
m2s = np.random.randint(30, 250, samples)
sales_price_bergen = np.zeros(samples, dtype=int)
sales_price_stavanger = np.zeros(samples, dtype=int)
sales_price_oslo = np.zeros(samples, dtype=int)
print(f" x0, y0 : {xs[0]}, {ys[0]}")
for i in range(samples) :
    sales_price_bergen[i] = house_price_bergen(xs[i], ys[i], house_types[i], standards[i], m2s[i]) * np.random.uniform(0.95, 1.05)
for i in range(samples) :
    sales_price_stavanger[i] = house_price_stavanger(xs[i], ys[i], house_types[i], standards[i], m2s[i]) * np.random.uniform(0.95, 1.05)
for i in range(samples) :
    sales_price_oslo[i] = house_price_oslo(xs[i], ys[i], house_types[i], standards[i], m2s[i]) * np.random.uniform(0.95, 1.05)

# DataFrame
df1 = pd.DataFrame({
    "x": xs,
    "y": ys,
    "m2": m2s,
    "house_type": house_types,
    "standard": np.round(standards, 1),
    "sales_price": sales_price_bergen
})

df2 = pd.DataFrame({
    "x": xs,
    "y": ys,
    "m2": m2s,
    "house_type": house_types,
    "standard": np.round(standards, 1),
    "sales_price": sales_price_stavanger
})

df3 = pd.DataFrame({
    "x": xs,
    "y": ys,
    "m2": m2s,
    "house_type": house_types,
    "standard": np.round(standards, 1),
    "sales_price": sales_price_oslo
})

df1.to_csv("House_Prices_Bergen.csv", index=False)
df2.to_csv("House_Prices_Stavanger.csv", index=False)
df3.to_csv("House_Prices_Oslo.csv", index=False)

# print(df.head())
print("CSV file 'House_Prices_Bergen.csv' created successfully.")
print(f"Sentrum (leil, 100m2, god std) = {house_price_bergen(40, 58, 3, 3, 100):>10,.0f}") # Sentrum !
print(f"Fana (leil, 100m2, god std)   = {house_price_bergen(40, 20, 3, 3, 100):>10,.0f}\n") # Sentrum !

print("CSV file 'House_Prices_Stavanger.csv' created successfully.")
print(f"Sentrum (leil, 100m2, god std) = {house_price_bergen(75, 75, 3, 3, 100):>10,.0f}") # Sentrum !
print(f"Sandnes (leil, 100m2, god std)   = {house_price_bergen(60, 30, 3, 3, 100):>10,.0f}\n") # Sentrum !

print("CSV file 'House_Prices_Oslo.csv' created successfully.")
print(f"Sentrum (leil, 100m2, god std) = {house_price_bergen(45, 60, 3, 3, 100):>10,.0f}") # Sentrum !
print(f"Skøyen (leil, 100m2, god std)   = {house_price_bergen(20, 60, 3, 3, 100):>10,.0f}\n") # Sentrum !