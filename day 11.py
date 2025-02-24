import numpy as np
import pandas as pd

df = pd.DataFrame(
    [[4, 7, 10],
     [5, 8, 11],
     [6, 9, 12]], index=[1, 2, 3], columns=['a', 'b', 'c']
)
print(df)