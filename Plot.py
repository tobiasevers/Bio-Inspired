import matplotlib.pyplot as plt
import pickle
import numpy as np
import scipy.stats as stats

#TODO: commenten
#TODO: functies overzichtelijk neerzetten
#TODO: laden van models overzichtelijk neerzetten
#TODO: main plot makkelijk werkbaar maken

### PLOT FUNCTIONS ###
def plot_heatmap(weighted_array, title, lst_close, lst_comm, text=True):
    """
    Plot a heatmap from a weighted array with labels and a color bar.
    """
    fig, ax = plt.subplots(figsize=(10, 8))  # Create a figure and axis for the plot
    cax = ax.matshow(weighted_array / 800, cmap='Blues', interpolation='nearest')  # Display the heatmap
    # Color bar with increased font size for label
    cbar = fig.colorbar(cax, fraction=0.046, pad=0.04)  # Add a color bar to the heatmap
    cbar.set_label('Error Rate', fontsize=14)  # Set the label for the color bar

    if text:
        # Annotate the heatmap with the values
        for i in range(weighted_array.shape[0]):
            for j in range(weighted_array.shape[1]):
                ax.text(j, i, f'{weighted_array[i, j]:.2f}', ha='center', va='center', color='black')

    plt.title(title)  # Set the title of the plot
    plt.xlabel('Close Range', fontsize=16)  # Label the x-axis
    plt.ylabel('Communication Range', fontsize=16)  # Label the y-axis
    ax.set_xticks(range(len(weighted_array[0])))  # Set x-tick positions
    ax.set_xticklabels(lst_close, fontsize=12)  # Set x-tick labels
    ax.set_yticks(range(len(weighted_array)))  # Set y-tick positions
    ax.set_yticklabels(lst_comm, fontsize=12)  # Set y-tick labels
    ax.tick_params(axis='x', bottom=True, top=False, labelbottom=True, labeltop=False)  # Configure tick parameters
    plt.show()  # Display the plot

def plot_speed(data):
    # Extract data for plotting
    comm_x = list(data['COMM'].keys())  # X-values for COMM data
    comm_y = list(data['COMM'].values())  # Y-values for COMM data
    nocomm_x = list(data['NOCOMM'].keys())  # X-values for NOCOMM data
    nocomm_y = list(data['NOCOMM'].values())  # Y-values for NOCOMM data

    # Plotting the data
    plt.figure(figsize=(10, 6))  # Create a figure for the plot
    plt.plot(comm_x, comm_y, label='COMM', marker='o', color='dodgerblue')  # Plot COMM data
    plt.plot(nocomm_x, nocomm_y, label='NOCOMM', marker='o', color='deepskyblue')  # Plot NOCOMM data

    plt.title('', fontsize=16)  # Set the title of the plot
    plt.xlabel('Speed Debris', fontsize=16)  # Label the x-axis
    plt.ylabel('Error Rate', fontsize=16)  # Label the y-axis
    plt.legend(fontsize=16)  # Add a legend to the plot
    plt.grid(True)  # Add a grid to the plot
    plt.show()  # Display the plot

def plot_multiple(results_dicts, close, lst_comm, degree=2, lst_label=['x', 'x', 'x', 'x', 'x'], poly=False):
    """
    Plot multiple result dictionaries with optional polynomial fitting.
    """
    plt.figure(figsize=(10, 6))  # Create a figure for the plot
    colors = ['dodgerblue', 'deepskyblue', 'steelblue', 'cornflowerblue', 'royalblue', 'red', 'green', 'black']  # Define colors for the plots

    for i, (result_dict, color) in enumerate(zip(results_dicts, colors)):
        lst = []
        for key in result_dict.keys():
            lst.append(result_dict[key][close])  # Collect data for the specific 'close' value

        # Convert lst_comm and lst to numpy arrays for fitting
        x = np.array(lst_comm)
        y = np.array(lst)

        if poly:
            # Fit the points with a polynomial of the given degree
            coeffs = np.polyfit(x, y, degree)
            poly_func = np.poly1d(coeffs)

            # Generate values for the fitted polynomial line
            x_fine = np.linspace(x.min(), x.max(), 500)
            y_fitted = poly_func(x_fine)

            # Plot original data points and fitted line
            plt.plot(x, y, 'o', color=color, label=lst_label[i])  # Original data points
            plt.plot(x_fine, y_fitted, '-', color=color)  # Fitted polynomial line
        else:
            plt.plot(x, y, 'o-', color=color, label=lst_label[i])  # Plot original data points with lines

    plt.title('', fontsize=16)  # Set the title of the plot
    plt.xlabel('Communication Range', fontsize=16)  # Label the x-axis
    plt.ylabel('Error Rate', fontsize=16)  # Label the y-axis
    plt.legend(fontsize=12, loc='best')  # Add a legend to the plot
    plt.grid(True)  # Add a grid to the plot
    plt.show()  # Display the plot

def divide_dict_values(d, divisor):
    """
    Recursively divide all values in the dictionary by the divisor.
    Handles nested dictionaries.
    """
    for key, value in d.items():
        if isinstance(value, dict):
            divide_dict_values(value, divisor)  # Recursively divide values in nested dictionaries
        else:
            d[key] = "SUCCESS" if value == 0 else value / divisor  # Divide value by divisor, handle zero case
    return d  # Return the updated dictionary

### LOAD DATA ###

# Load the dictionary from the file for the contour plot with communication
with open('Results/Contour_model3_50_200_0_150_500_0_comm.pkl', 'rb') as file:
    lst_comm = range(50,200,10)
    lst_close = range(0,150,10)
    # Convert the dictionary to a 2D array for plotting
    comm_values = sorted(lst_comm)[:]
    close_values = sorted(lst_close)[:]
    dict_results_500_0_norm = pickle.load(file)
    Z1 = np.zeros((len(comm_values), len(close_values)))
    for i, comm in enumerate(comm_values):
        for j, close in enumerate(close_values):
            Z1[i, j] = dict_results_500_0_norm[comm][close]

# Load the dictionary from the file for the contour plot without communication
with open('Results/Contour_model3_50_200_0_150_500_0_nocomm.pkl', 'rb') as file:
    lst_comm = range(50,200,10)
    lst_close = range(0,150,10)
    # Convert the dictionary to a 2D array for plotting
    comm_values = sorted(lst_comm)[:]
    close_values = sorted(lst_close)[:]
    dict_results2 = pickle.load(file)
    Z2 = np.zeros((len(comm_values), len(close_values)))
    for i, comm in enumerate(comm_values):
        for j, close in enumerate(close_values):
            Z2[i, j] = dict_results2[comm][close]


with open('Results/Contour_model3_150_80_speedcomm_1_10_norm.pkl', 'rb') as file:
    dict_results2 = pickle.load(file)

with open('Results/Contour_model3_50_200_0_150_300_0_comm.pkl', 'rb') as file:
    dict_results_300_0_norm = pickle.load(file)
with open('Results/Contour_model3_50_200_0_150_300_800_comm.pkl', 'rb') as file:
    dict_results_300_800_norm = pickle.load(file)
with open('Results/Contour_model3_50_200_0_150_500_800_comm.pkl', 'rb') as file:
    dict_results_500_800_norm = pickle.load(file)

with open('Results/Contour_model3_50_200_50_500_0_comm_multi_vertical_all.pkl', 'rb') as file:
    dict_results_500_0_multi = divide_dict_values(pickle.load(file), 1/800)

with open('Results/Contour_model3_50_200_50_500_0_comm_multi_horizontal.pkl', 'rb') as file:
    dict_results_500_0_multi_hori = divide_dict_values(pickle.load(file), 1/800)

with open('Results/Contour_model3_50_200_50_500_0_comm_multi_vertical_upper_lr.pkl', 'rb') as file:
    dict_results_500_0_multi_veri_upper_lr = divide_dict_values(pickle.load(file), 1/800)

with open('Results/Contour_model3_50_200_50_500_0_comm_multi_vertical_upper_rl.pkl', 'rb') as file:
    dict_results_500_0_multi_veri_upper_rl = divide_dict_values(pickle.load(file), 1/800)

with open('Results/Contour_model3_50_200_80_300_0_comm_multitrain.pkl', 'rb') as file:
    dict_results_300_0_multi = pickle.load(file)

with open('Results/Contour_model3_50_200_80_300_800_comm_multitrain.pkl', 'rb') as file:
    dict_results_300_800_multi = pickle.load(file)

with open('Results/Contour_model3_50_200_80_500_800_comm_multitrain.pkl', 'rb') as file:
    dict_results_500_800_multi = pickle.load(file)

with open('Results/Contour_model3_50_200_80_300_0_comm_upperleft.pkl', 'rb') as file:
    dict_results_300_0_upperleft = pickle.load(file)

with open('Results/Contour_model3_50_200_80_300_0_comm_upperleft_2.pkl', 'rb') as file:
    dict_results_300_0_upperleft_2 = pickle.load(file)

with open('Results/Contour_model3_50_200_80_500_0_comm_upperright_05.pkl', 'rb') as file:
    dict_results_500_0_upperright_05 = pickle.load(file)

with open('Results/Contour_model3_50_200_80_500_0_comm_upperright_025.pkl', 'rb') as file:
    dict_results_500_0_upperright_025 = pickle.load(file)

with open('Results/Contour_model3_50_200_80_500_0_comm_upperright_075.pkl', 'rb') as file:
    dict_results_500_0_upperright_075 = pickle.load(file)

with open('Results/Contour_model3_50_200_80_500_0_comm_upperright_orig.pkl', 'rb') as file:
    dict_results_500_0_upperright_orig = pickle.load(file)

with open('Results/Contour_model3_50_200_80_500_0_comm_upperright_090.pkl', 'rb') as file:
    dict_results_500_0_upperright_090 = pickle.load(file)

with open('Results/Contour_model3_50_200_80_500_0_comm_upperright_080.pkl', 'rb') as file:
    dict_results_500_0_upperright_080 = pickle.load(file)

with open('Results/Contour_model3_50_200_80_500_0_comm_upperright_1.pkl', 'rb') as file:
    dict_results_500_0_upperright_100 = pickle.load(file)

with open('Results/Contour_model3_120_50_angles_comm_upperright.pkl', 'rb') as file:
    dict_results_500_0_upperright_angles = pickle.load(file)

with open('Results/Contour_model3_50_200_50_80_500_0_comm_upperright_epi_25.pkl', 'rb') as file:
    dict_results_500_0_upperright_epi25 = pickle.load(file)

with open('Results/Contour_model3_50_200_50_80_500_0_comm_upperright_epi_50.pkl', 'rb') as file:
    dict_results_500_0_upperright_epi50 = pickle.load(file)

with open('Results/Contour_model3_50_200_50_80_500_0_comm_upperright_epi_75.pkl', 'rb') as file:
    dict_results_500_0_upperright_epi75 = pickle.load(file)

with open('Results/Contour_model3_50_200_50_80_500_0_comm_upperright_epi_100.pkl', 'rb') as file:
    dict_results_500_0_upperright_epi100 = pickle.load(file)

with open('Results/Contour_model3_50_200_50_80_500_0_comm_upperright_epi_125.pkl', 'rb') as file:
    dict_results_500_0_upperright_epi125 = pickle.load(file)

with open('Results/Contour_model3_50_200_50_80_500_0_comm_upperright_epi_150.pkl', 'rb') as file:
    dict_results_500_0_upperright_epi150 = pickle.load(file)

with open('Results/Contour_model3_50_200_50_80_500_0_comm_upperright_epi_175.pkl', 'rb') as file:
    dict_results_500_0_upperright_epi175 = pickle.load(file)

with open('Results/Contour_model3_50_200_50_80_500_0_comm_upperright_epi_200.pkl', 'rb') as file:
     dict_results_500_0_upperright_epi200 = pickle.load(file)

with open('Results/Contour_model3_50_200_50_80_500_0_comm_upperright_epi_25_2.pkl', 'rb') as file:
    dict_results_500_0_upperright_epi25_2 = pickle.load(file)

with open('Results/Contour_model3_50_200_50_80_500_0_comm_upperright_epi_50_2.pkl', 'rb') as file:
     dict_results_500_0_upperright_epi50_2 = pickle.load(file)

with open('Results/Contour_model3_50_200_50_80_500_0_comm_upperright_epi_75_2.pkl', 'rb') as file:
    dict_results_500_0_upperright_epi75_2 = pickle.load(file)

with open('Results/Contour_model3_50_200_50_80_500_0_comm_upperright_epi_100_2.pkl', 'rb') as file:
    dict_results_500_0_upperright_epi100_2 = pickle.load(file)

with open('Results/Contour_model3_50_200_50_80_500_0_comm_upperright_epi_125_2.pkl', 'rb') as file:
    dict_results_500_0_upperright_epi125_2 = pickle.load(file)

with open('Results/Contour_model3_50_200_50_80_500_0_comm_upperright_epi_150_2.pkl', 'rb') as file:
    dict_results_500_0_upperright_epi150_2 = pickle.load(file)

with open('Results/Contour_model3_50_200_50_80_500_0_comm_upperright_epi_175_2.pkl', 'rb') as file:
    dict_results_500_0_upperright_epi175_2 = pickle.load(file)

with open('Results/Contour_model3_50_200_50_80_500_0_comm_upperright_epi_200_2.pkl', 'rb') as file:
    dict_results_500_0_upperright_epi200_2 = pickle.load(file)

if __name__ == '__main__':
    ## PLOTS FOR COMMUNICATION AND CLOSE RANGE
    # Heatmap for varying communication and range perimeters
    plot_heatmap(Z1, '', lst_close[:], lst_comm[:], text=False)  # Heatmap with communication
    plot_heatmap(Z2, '', lst_close[:], lst_comm[:], text=False)  # Heatmap without communication

    ## PLOTS FOR VARYING SPEED AND COMMUNICATION ABILITY, 150 80
    plot_speed(dict_results2)  # Plot speed data
    comm = list(dict_results2['COMM'].values())[-4:]  # Last 4 COMM results
    nocomm = list(dict_results2['NOCOMM'].values())[-4:]  # Last 4 NOCOMM results
    teststat, p_value = stats.wilcoxon(comm, nocomm)  # Wilcoxon test
    print('P_value for comm, no comm speed 3-6:', p_value)

    ## PLOTS FOR DIFFERENT DEBRIS TRAINING ORIGINS 50, 100 (normal, (verti upper), vertical and horizontal)
    plot_multiple([dict_results_500_0_norm, dict_results_500_0_multi_veri_upper_lr, dict_results_500_0_multi_veri_upper_rl, dict_results_500_0_multi_hori, dict_results_500_0_multi], 50, lst_comm=range(50, 200, 10), degree=3,
                  lst_label=['normal', 'vertical LR', 'vertical RL', 'multi horizontal', 'multi vertical'], poly=False)  # Plot multiple configurations
    resLR = [inner_dict[50] for inner_dict in dict_results_500_0_multi_veri_upper_lr.values()][-10:]  # Last 10 results for vertical LR
    resRL = [inner_dict[50] for inner_dict in dict_results_500_0_multi_veri_upper_rl.values()][-10:]  # Last 10 results for vertical RL
    teststat, p_value = stats.wilcoxon(resLR, resRL)  # Wilcoxon test
    print('P_value for RL, LR:', p_value)

    ## PLOTS FOR DIFFERENT EPSILON
    plot_multiple([dict_results_500_0_upperright_025, dict_results_500_0_upperright_05, dict_results_500_0_upperright_075, dict_results_500_0_upperright_100]
                   , 80, lst_comm=range(50,200,10),
                   degree=3, lst_label=[0.25, 0.50, 0.75, 1.0, 'multi'], poly=False)  # Plot multiple epsilon configurations
    res025 = [inner_dict[80] for inner_dict in dict_results_500_0_upperright_025.values()][-6:]  # Last 6 results for epsilon 0.25
    res050 = [inner_dict[80] for inner_dict in dict_results_500_0_upperright_05.values()][-6:]  # Last 6 results for epsilon 0.50
    res075 = [inner_dict[80] for inner_dict in dict_results_500_0_upperright_075.values()][-6:]  # Last 6 results for epsilon 0.75
    res100 = [inner_dict[80] for inner_dict in dict_results_500_0_upperright_100.values()][-6:]  # Last 6 results for epsilon 1.0
    teststat, p_value1 = stats.wilcoxon(res025, res075)  # Wilcoxon test between 0.25 and 0.75
    teststat, p_value2 = stats.wilcoxon(res025, res100)  # Wilcoxon test between 0.25 and 1.0
    teststat, p_value3 = stats.wilcoxon(res075, res100)  # Wilcoxon test between 0.75 and 1.0
    print('P_values for 25-75, 25-100 and 75-100:', p_value1, p_value2, p_value3)

    # Different angles and speeds results
    print('The results for different angles and speeds of incoming debris:')
    print(dict_results_500_0_upperright_angles)

    ## PLOTS FOR DIFFERENT EPISODES
    plot_multiple(
        [dict_results_500_0_upperright_epi25, dict_results_500_0_upperright_epi50, dict_results_500_0_upperright_epi75,
         dict_results_500_0_upperright_epi100, dict_results_500_0_upperright_epi125,
         dict_results_500_0_upperright_epi150, dict_results_500_0_upperright_epi175,
         dict_results_500_0_upperright_epi200],
        70, lst_comm=range(50, 200, 10), degree=3,
        lst_label=['epi25', 'epi50', 'epi75', 'epi100', 'epi125', 'epi150', 'epi175', 'epi200'], poly=False)  # Plot results for different episodes

    ## PLOT FOR DIFFERENCE IN TRAINING PARAMETERS 50 EPISODES
    plot_multiple([dict_results_500_0_upperright_epi50, dict_results_500_0_upperright_epi50_2], 50, lst_comm=range(50, 200, 10), degree=3,
        lst_label=['170-50', '150-80'], poly=False)  # Plot differences in training parameters
    res1 = [inner_dict[50] for inner_dict in dict_results_500_0_upperright_epi50.values()]  # Results for training parameter set 1
    res2 = [inner_dict[50] / 800 for inner_dict in dict_results_500_0_upperright_epi50_2.values()]  # Results for training parameter set 2
    teststat, p_value = stats.wilcoxon(res1, res2)  # Wilcoxon test
    print('P_value for different training configurations:', p_value)





