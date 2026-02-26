import pandas as pd
import matplotlib.pyplot as plt
import io

data = """Framework,Num_Procs,Epoch_Time_s
Julia,1,0.7667
Julia,2,0.7820
Julia,4,0.8193
Python,1,1.8136
Python,2,1.0986
Python,4,0.7965
"""

df = pd.read_csv(io.StringIO(data))
df['Throughput'] = (df['Num_Procs'] * 24) / df['Epoch_Time_s']

julia_df = df[df['Framework'] == 'Julia']
python_df = df[df['Framework'] == 'Python']

plt.figure(figsize=(10, 6))
plt.plot(julia_df['Num_Procs'], julia_df['Throughput'], marker='o', label='Julia (Lux.jl)')
plt.plot(python_df['Num_Procs'], python_df['Throughput'], marker='s', label='Python (PyTorch)')

plt.xlabel('Number of GPUs')
plt.ylabel('Throughput (Samples/s)')
plt.title('Strong Scaling on NVIDIA H100 (ResNet-152)')
plt.legend()
plt.grid(True)
plt.xticks([1, 2, 4])
plt.savefig('grant_proposal/julich/scaling_plot.png')
print("Plot saved to grant_proposal/julich/scaling_plot.png")
