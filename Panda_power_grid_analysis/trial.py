import pandapower as pp
import pandapower.networks as nw
import pandapower.plotting as plot
from simbench import get_simbench_net
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the SimBench rural low-voltage grid 3 network
simbench_code = "1-LV-rural1--0-no_sw"
net = get_simbench_net(simbench_code)

#pp.runpp(net)
#print(net.bus)

grid_bus_index = 42

load_values = [10, 15, 20, 10, 12, 60, 80, 80, 25, 12, 30, 15, 22]  # kW values
load_values = np.array(load_values)

for i, p_kw in enumerate(load_values):
    # Convert kW to MW and update the load
    net.load.loc[i, "p_mw"] = p_kw / 1000  # Convert kW to MW
    net.load.loc[i, "q_mvar"] = 0  # Reactive power set to 0

net.bus.loc[0:12, "vn_kv"] = 0.4  # Update LV buses to 0.4 kV
net.bus.loc[grid_bus_index, "vn_kv"] = 20.0  # Update the grid bus to 20 kV
net.bus.loc[grid_bus_index, "s_sc_max_mva"] = 0.2  # Max short-circuit power in MVA

pp.runpp(net)

cmap_list=[(0, "green"), (50, "yellow"), (100, "red")]
cmap, norm = plot.cmap_continuous(cmap_list)
lc = plot.create_line_collection(net, net.line.index, zorder=1, cmap=cmap, norm=norm, linewidths=2)
plot.draw_collections([lc], figsize=(8,6))

#plt.savefig('voltage.png')


#print(net.bus)
#print('#################################################################################################################')
#print(net.line)
#print('#################################################################################################################')
#print(net.trafo)
#print('#################################################################################################################')
#print(net.load)
#print('#################################################################################################################')

