import numpy as np
a = np.arange(12).reshape(3, 4)
print(np.delete(a,[1,2,0],0))
