Average_SOC_fucntion = [Average_SOC_fucntion(csv_file_path_Actual),
                    Average_SOC_fucntion(csv_file_path_10),
                    Average_SOC_fucntion(csv_file_path_20),
                    Average_SOC_fucntion(csv_file_path_30),
                    Average_SOC_fucntion(csv_file_path_40),
                    Average_SOC_fucntion(csv_file_path_50),
                    Average_SOC_fucntion(csv_file_path_60),
                    Average_SOC_fucntion(csv_file_path_70),
                    Average_SOC_fucntion(csv_file_path_80),
                    Average_SOC_fucntion(csv_file_path_90),
                    Average_SOC_fucntion(csv_file_path_100),
                    ]

labels = ['0', '10', '20', '30', '40', '50', '60', '70', '80', '90', '100']

fig1, ax1 = plt.subplots()
ax1.bar(labels, Average_SOC_fucntion)
ax1.set_title('Self Consumption ')
ax1.set_xlabel('NILM Efficiency (%)')
ax1.set_ylabel('Self Consumption (%)')
plt.grid()
fig1.savefig(FILE_DIR_PATH/f'NILM_and_DRL_based_EMS/NILM_model/NILM_efficiency/Figures/Average_SOC_fucntion.png')