
import math
import pandapower as pp
import pandapower.networks as nw
import pandapower.plotting as plot
from simbench import get_simbench_net
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandapower import timeseries
from pandapower import control

simbench_code = "1-LV-rural1--0-no_sw"
net = get_simbench_net(simbench_code)

SCP = 0.2

net.ext_grid.loc[net.ext_grid['bus'] == 42, 'vm_pu'] = 1


Resistance = ( 400 ** 2 ) / ( ( SCP * 1e6 ) * math.sqrt( 1 + ( 1 / 2.5 ) ** 2 ) )
Impedance = Resistance / ( 2.5 * 2 * np.pi * 50 )
Reactance = 2 * np.pi * Impedance


line_params = {
    "from_bus": 42,  # High-voltage side of the transformer (shared with external grid)
    "to_bus": 3,  # Low-voltage side of the transformer
    "length_km": 5,  # Length of the line
    "name": "ExternalGrid-TrafoLine",
    "r_ohm_per_km": Resistance,  # Resistance per km
    "x_ohm_per_km": Reactance,  # Reactance per km
    "c_nf_per_km": 10,  # Capacitance per km
    "max_i_ka": 1  # Maximum current capacity
}

line_idx = pp.create_line_from_parameters(net, **line_params)

#fig, ax = plt.subplots(figsize=(6.4,4.8))
#plot.simple_plot(net, show_plot=False)
#plt.savefig(f'Figures/Grid_Model_1-LV-rural1--0-no_sw.png')


#profile = 'Residual_Actual'
profile = 'Residual_Optimized'

df = pd.read_csv(f'Database/{profile}.csv',sep=';')
df = df / 1000000
ds = timeseries.DFData(df)

const_load = control.ConstControl(
    net,
    element='load',
    element_index=net.load.index,
    variable='p_mw',
    data_source=ds,
    profile_name=[f"Load_{i}" for i in range(len(net.load.index))]
)

ow = timeseries.OutputWriter(net, output_path="./", output_file_type=".csv")
ow.log_variable('res_bus', f'vm_pu')
ow.log_variable('res_line', f'loading_percent')

timeseries.run_timeseries(net, time_steps=(47,0))

cmap_list_lc = [(0, "green"), (60, "yellow"), (120, "red")]
cmap_lc, norm_lc = plot.cmap_continuous(cmap_list_lc)
lc = plot.create_line_collection(net, net.line.index, zorder=1, cmap=cmap_lc, norm=norm_lc, linewidths=2)

cmap_list_bc = [(0.95, "red"), (0.975, "yellow"), (1.00, "green"), (1.025, "yellow"), (1.05, "red")]
cmap_bc, norm_bc = plot.cmap_continuous(cmap_list_bc)
bc = plot.create_bus_collection(net, net.bus.index, size=5/100000, zorder=1, cmap=cmap_bc, norm=norm_bc)

plot.draw_collections([lc, bc], figsize=(6.4,4.8))
plt.savefig(f'Figures/{profile}_{SCP}.svg')
