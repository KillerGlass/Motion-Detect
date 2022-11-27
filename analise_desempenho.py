import pandas as pd
import matplotlib.pyplot as plt

av = pd.read_csv('reporter.csv')

res_av = av.groupby(['Frame']).sum()
print(res_av)


res_av.plot.bar()
plt.show()
#plt.savefig('bar.png',img) /     