import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
script_dir = os.path.dirname(os.path.abspath(__file__))
"""
    This script plots the costs of the reconstructions
"""
def plot_cost(totalCosts, dataAttcosts, regCosts, times, file_name, title):

    totalCosts_initial = totalCosts[0,:]
    totalCosts_astra_matrix = totalCosts[1,:]
    totalCosts_astra = totalCosts[1,:]

    regCosts_initial = regCosts[0,:]
    regCosts_astra_matrix = regCosts[1,:]
    regCosts_astra = regCosts[1,:]

    dataAttCosts_initial = dataAttcosts[0,:]
    dataAttCosts_astra_matrix = dataAttcosts[1,:]
    dataAttCosts_astra = dataAttcosts[1,:]

    times_initial = times[0,:]
    times_astra_matrix = times[1,:]
    times_astra = times[1,:]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    

    plt.title(title)

    axes[0].plot(totalCosts_initial, label='Initial Projector', linestyle='--')
    axes[0].plot(totalCosts_astra, label='ASTRA Projector', alpha=0.7)
    # axes[0].plot(totalCosts_astra_matrix, label='ASTRA Matrix', alpha = 0.9)
    axes[0].set_xlabel('Iteration')
    axes[0].set_ylabel('TotalCost')
    axes[0].set_title('Convergence Comparison Total Costs')
    axes[0].legend()



    axes[1].plot(regCosts_initial, label='Initial Projector', linestyle='--')
    axes[1].plot(regCosts_astra, label='ASTRA Projector', alpha=0.7)
    # axes[1].plot(regCosts_astra_matrix, label='ASTRA Matrix', alpha=0.9)
    axes[1].set_xlabel('Iteration')
    axes[1].set_ylabel('RegCost')
    axes[1].set_title('Convergence Comparison regCosts')
    axes[1].legend()



    axes[2].plot(dataAttCosts_initial, label='Initial Projector', linestyle='--')
    axes[2].plot(dataAttCosts_astra, label='ASTRA Projector', alpha=0.7)
    # axes[2].plot(dataAttCosts_astra_matrix, label='ASTRA Matrix', alpha=0.9)
    axes[2].set_xlabel('Iteration')
    axes[2].set_ylabel('DataAttCost')
    axes[2].set_title('Convergence Comparison dataAttCosts')
    axes[2].legend()
    plt.tight_layout()
    plt.savefig(f'code_simple_water_bone/costs/costs_vs_iteration_{file_name}.png')
    plt.show()


    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    plt.title(title)


    axes[0].plot(times_initial, totalCosts_initial, label='Initial Projector', linestyle='--')
    axes[0].plot(times_astra, totalCosts_astra, label='ASTRA Projector', alpha=0.7)
    # axes[0].plot(times_astra_matrix, totalCosts_astra_matrix, label='ASTRA Matrix', alpha = 0.9)
    axes[0].set_xlabel('TotalCost')
    axes[0].set_title('Convergence Comparison Total Costs as a function of Time')
    axes[0].legend()



    axes[1].plot(times_initial, regCosts_initial, label='Initial Projector', linestyle='--')
    axes[1].plot(times_astra, regCosts_astra, label='ASTRA Projector', alpha=0.7)
    # axes[1].plot(times_astra_matrix, regCosts_astra_matrix, label='ASTRA Matrix', alpha=0.9)
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('RegCost')
    axes[1].set_title('Convergence Comparison regCosts as a function of Time')
    axes[1].legend()



    axes[2].plot(times_initial, dataAttCosts_initial, label='Initial Projector', linestyle='--')
    axes[2].plot(times_astra, dataAttCosts_astra, label='ASTRA Projector', alpha=0.7)
    # axes[2].plot(times_astra_matrix, dataAttCosts_astra_matrix, label='ASTRA Matrix', alpha=0.9)
    axes[2].set_xlabel('Time (s)')
    axes[2].set_ylabel('TataAttCost')
    axes[2].set_title('Convergence Comparison dataAttCosts as a function of Time')
    axes[2].legend()
    plt.tight_layout()

    plt.savefig(f'code_simple_water_bone/costs/costs_vs_time_{file_name}.png')
    plt.show()

def plot_cost_npy():
    # Object Size a square
    objectSize = 32
    #
    # Number of iterations
    nIter = 10

    # Number of subsets
    nSubset = 4

    # Number of pixels on the detector plate.

    nPixelsY = 128
    nPixelsZ = 64

    # Pixel pitch
    pixelPitch = 1

    d_detector = 900
    dSource = 2300

    n_proj = 150
    projAngles = np.arange(0, n_proj) * (np.pi / n_proj)

    # The total number of pixels on the detector plate
    nPixelsPerProj = nPixelsZ * nPixelsY

    astra = 1  # 0: conebeamtomo3, 1: astra matrix, 2: astra projector
    save = True  # save plots
    scale = 0.1  # do not touch, this is to adjust the astra matrix and projector to work with attenuation factors etc
    idx_phantom = 0  # select which phantom geometry you want
    noise = True
    # # Set regularization parameters
    lambda_ = np.array([10, 5])  # From [100, 50]
    delta_huber = np.array([0.01, 0.05])  # [0.01, 0.05]
    current_dir = Path(__file__).parent

    if noise:
        directory = f"npy_files_noise_idx{idx_phantom}"
    else:
        directory = f"npy_files_idx{idx_phantom}"

    title = run_name = "mulptiple_materials_nProj_50"  # change name for file based on what you are running eg. original matrix vs astra projector
    file_name = f"{run_name}_{objectSize}_phantom{idx_phantom}_{nIter}_{nSubset}_{scale}_{lambda_}"
    title = f"nIter = {nIter}, nSubset = {nSubset}, scale = {scale}, lambda_ = {lambda_}"

    totalCost_path_0 =current_dir.parent / "code_simple_water_bone" / directory / f'totalCost_0_{nIter}_{nSubset}_{objectSize}_{nPixelsY}_{nPixelsZ}_{lambda_}.npy'
    dataAttCost_path_0 = current_dir.parent / "code_simple_water_bone" / directory / f'dataAttCost_0_{nIter}_{nSubset}_{objectSize}_{nPixelsY}_{nPixelsZ}_{lambda_}.npy'
    regCost_path_0 = current_dir.parent / "code_simple_water_bone" / directory / f'regAttCost_0_{nIter}_{nSubset}_{objectSize}_{nPixelsY}_{nPixelsZ}_{lambda_}.npy'
    times_path_0 = current_dir.parent / "code_simple_water_bone" / directory / f'times_0_{nIter}_{nSubset}_{objectSize}_{nPixelsY}_{nPixelsZ}_{lambda_}.npy'

    totalCost_path_1 =current_dir.parent / "code_simple_water_bone" / directory / f'totalCost_1_{nIter}_{nSubset}_{objectSize}_{nPixelsY}_{nPixelsZ}_{lambda_}.npy'
    dataAttCost_path_1 = current_dir.parent / "code_simple_water_bone" / directory / f'dataAttCost_1_{nIter}_{nSubset}_{objectSize}_{nPixelsY}_{nPixelsZ}_{lambda_}.npy'
    regCost_path_1 = current_dir.parent / "code_simple_water_bone" / directory / f'regAttCost_1_{nIter}_{nSubset}_{objectSize}_{nPixelsY}_{nPixelsZ}_{lambda_}.npy'
    times_path_1 = current_dir.parent / "code_simple_water_bone" / directory / f'times_1_{nIter}_{nSubset}_{objectSize}_{nPixelsY}_{nPixelsZ}_{lambda_}.npy'

    totalCosts_initial = np.load(str(totalCost_path_0))
    totalCosts_astra = np.load(str(totalCost_path_1))

    regCosts_initial = np.load(str(regCost_path_0))
    regCosts_astra = np.load(str(regCost_path_1))

    dataAttCosts_initial = np.load(str(dataAttCost_path_0))
    dataAttCosts_astra = np.load(str(dataAttCost_path_1))

    times_initial = np.load(str(times_path_0))
    times_astra = np.load(str(times_path_1))

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    plt.title(title)

    axes[0].plot(totalCosts_initial, label='Initial Projector', linestyle='--')
    axes[0].plot(totalCosts_astra, label='ASTRA Projector', alpha=0.7)
    axes[0].set_xlabel('Iteration')
    axes[0].set_ylabel('TotalCost')
    axes[0].set_title('Convergence Comparison Total Costs')
    axes[0].legend()

    axes[1].plot(regCosts_initial, label='Initial Projector', linestyle='--')
    axes[1].plot(regCosts_astra, label='ASTRA Projector', alpha=0.7)
    axes[1].set_xlabel('Iteration')
    axes[1].set_ylabel('RegCost')
    axes[1].set_title('Convergence Comparison regCosts')
    axes[1].legend()

    axes[2].plot(dataAttCosts_initial, label='Initial Projector', linestyle='--')
    axes[2].plot(dataAttCosts_astra, label='ASTRA Projector', alpha=0.7)
    # axes[2].plot(dataAttCosts_astra_matrix, label='ASTRA Matrix', alpha=0.9)
    axes[2].set_xlabel('Iteration')
    axes[2].set_ylabel('DataAttCost')
    axes[2].set_title('Convergence Comparison dataAttCosts')
    axes[2].legend()
    plt.tight_layout()

    iterations_plot_path = current_dir.parent / "Python code for multiple materials - 12 mei 2025" / "more_materials_cost" / f'costs_vs_iteration_{file_name}.png'
    plt.savefig(str(iterations_plot_path))
    plt.show()

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    plt.title(title)

    axes[0].plot(times_initial, totalCosts_initial, label='Initial Projector', linestyle='--')
    axes[0].plot(times_astra, totalCosts_astra, label='ASTRA Projector', alpha=0.7)
    # axes[0].plot(times_astra_matrix, totalCosts_astra_matrix, label='ASTRA Matrix', alpha = 0.9)
    axes[0].set_xlabel('TotalCost')
    axes[0].set_title('Convergence Comparison Total Costs as a function of Time')
    axes[0].legend()

    axes[1].plot(times_initial, regCosts_initial, label='Initial Projector', linestyle='--')
    axes[1].plot(times_astra, regCosts_astra, label='ASTRA Projector', alpha=0.7)
    # axes[1].plot(times_astra_matrix, regCosts_astra_matrix, label='ASTRA Matrix', alpha=0.9)
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('RegCost')
    axes[1].set_title('Convergence Comparison regCosts as a function of Time')
    axes[1].legend()

    axes[2].plot(times_initial, dataAttCosts_initial, label='Initial Projector', linestyle='--')
    axes[2].plot(times_astra, dataAttCosts_astra, label='ASTRA Projector', alpha=0.7)
    # axes[2].plot(times_astra_matrix, dataAttCosts_astra_matrix, label='ASTRA Matrix', alpha=0.9)
    axes[2].set_xlabel('Time (s)')
    axes[2].set_ylabel('TataAttCost')
    axes[2].set_title('Convergence Comparison dataAttCosts as a function of Time')
    axes[2].legend()
    plt.tight_layout()

    time_plot_path = current_dir.parent / "Python code for multiple materials - 12 mei 2025" / "more_materials_cost" / f'costs_vs_time_{file_name}.png'
    plt.savefig(str(time_plot_path))
    plt.show()

if __name__ == '__main__':
    plot_cost_npy()