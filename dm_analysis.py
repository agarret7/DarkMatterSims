from math import *
import os
import warnings

import h5py
import matplotlib.pyplot as plt
import numpy as np

from matplotlib.colors import LogNorm
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import curve_fit

class Analyzer():
    """ Utility class for Analyzing data for Dark Matter Simulations. """

    def __init__(self, test_dir, test, timeslice, overwriting = False):
        """
        Constructor for the Analyzer object.
        
        :param test_dir: The directory storing the tests.
        :param test: The name of the test to analyze.
        :param timeslice: The timeslice of the data
        :param overwriting: Boolean to determine whether or not to overwrite previous analysis data.
        """
        self.file = h5py.File(test_dir + test + '/output/snap_' + timeslice + '.hdf5', 'r')
        self.timeslice = timeslice

        self.output_file = test_dir + '/analysis/' + test
        self.overwriting = False

        print("Made analyzer on file " + test)

    def print_directory(self, file = ''):
        """
        Prints the contents of an h5py directory structure at some specified location.
        :param directory: The location of the directory to print out.
        """
        print("File System:")
        if directory == '':
            for value in self.file.values():
                print(value)
        else:
            for value in self.file[directory].values():
                print(value)
        print('')

    def get_values(self, dataset):
        """
        Read h5py values into a numpy array.
        :param dataset: Dataset in h5py directory structure to parse into a numpy array.
        :return: Numpy array containing data at specified location.
        """
        arr = np.zeros(self.file["PartType1/" + dataset].shape)
        self.file["PartType1/" + dataset].read_direct(arr)
        return arr

    def make_magnitudes(self, data):
        """
        Takes a list of vectors and returns a list of their norms.
        :param data: Numpy array of shape (num_vectors, len_vectors)
        :return: Numpy array, mags, of shape (num_vectors) where mags[i] is the norm of data[i].
        """
        return np.apply_along_axis(np.linalg.norm, 1, data)

    def get_density(self, sample_bins):
        """
        TODO: Documentation
        """
        bins = sample_bins[:-1]

        self.SIDM_densities = np.sum(self.get_values("SIDM_Density"), 1)
        densities = np.zeros(len(bins))

        for i, r_edge in enumerate(bins):
            # For every edge in our bins...
            conditions = np.array([r_edge <= r < sample_bins[i+1] for r in self.distances])

            # Make a list of particles in between the edges.
            select_densities = self.SIDM_densities[conditions]

            # Calculate the Velocity Dispersion and Beta Anisotropy.
            densities[i] = np.median(select_densities)

        return densities

    def center(self, data):
        """
        Takes a list of data and shifts them so that the origin is at center of mass.
        :param data: TODO: What is the shape of this data?
        :return: data shifted such that the origin is at the center of mass.
        """
        try:
            # If the data are a list of scalars...
            shape = data.shape[1]
        except IndexError:
            # Our center is a scalar too.
            shape = 1
            
        center = np.zeros(shape, dtype = np.float64)
        
        for i, r in enumerate(data):
            center += self.masses[i] * r

        center_of_mass = center / sum(self.masses)
        shifted_arr = np.zeros(data.shape)

        if shape != 1:
            for i, n in enumerate(data):
                for j in range(shape):
                    shifted_arr[i][j] = n[j] - center_of_mass[j]
        else:
            for i, n in enumerate(data):
                shifted_arr[i] = n - center_of_mass

        return shifted_arr

    def make_hist(self, a, bins = 'auto', range = None, density = None):
        """
        Makes a numpy histogram with equal bins and frequencies.
        TODO: More documentation
        """
        H, bins = np.histogram(a, bins = bins, range = range, density = density)
        edges = bins[:-1]

        return np.array([edges, H])

    def make_densities(self, bins = 50, range = None, density = False):
        """
        Makes a special density histogram.
        TODO: More documentation
        """
        H, bins = np.histogram(self.distances, bins = bins, range = range, weights = self.masses, density = density)
        edges = bins[:-1]

        # Calculate the volume of each shell bin.
        volumes = np.array([((4/3)*pi*((r + bins[i+1])**3 - r**3)) for i, r in enumerate(edges)])

        return np.array([edges, H / volumes])

    def make_spherical_velocities(self):
        """
        Creates radial and tangential components of velocity from a set of particles' positions and velocities.
        TODO: More documentation
        """
        rad_components = tan_components = np.zeros(self.displacements.shape[0])
        
        for i, x in enumerate(self.displacements):
            # Normal vector to the of radius norm(x) is x / norm(x).
            norm_vec = x / np.linalg.norm(x)

            # Using dot product and subtracting radial component to find rad and tan components respectively.
            rad_components[i] = np.dot(self.velocities[i], norm_vec)
            tan_components[i] = np.linalg.norm(self.velocities[i] - rad_components[i] * norm_vec)

        return rad_components, tan_components

    def get_v0_beta(self, sample_bins, verbose = True):
        """
        TODO: Documentation
        """
        # Suppressing warnings for finding undefined variance.
        print("Making v0 and betas...")

        bins = sample_bins[:-1]

        print("Total bins: ", len(bins))

        np.seterr(invalid = 'ignore')
        warnings.filterwarnings("ignore", "Degrees of freedom <= 0 for slice", RuntimeWarning)

        rad_components, tan_components = self.make_spherical_velocities()

        v0 = betas = np.zeros(len(bins))

        for i, r_edge in enumerate(bins):
            print("bin", i, "...", end = ' ')
            
            # For every edge in our bins...
            conditions = np.array([r_edge <= r < sample_bins[i+1] for r in self.distances])

            # Make a list of particles in between the edges.
            select_rad_components = rad_components[conditions]
            select_tan_components = tan_components[conditions]
            select_speeds = self.speeds[conditions]

            # Calculate the Velocity Dispersion and Beta Anisotropy.
            v0[i] = np.std(select_speeds)
            betas[i] = 1 - np.var(select_rad_components) / np.var(select_tan_components)

            print("done")

        warnings.resetwarnings()

        return v0, betas

    def density_2d(self, plotting = True, verbose = True):
        """
        TODO: Documentation
        """

        if verbose: print("2d mass density plot:")
              
        self.displacements = self.get_values("Coordinates")
        self.masses = self.get_values("Masses")

        if verbose: print("Made masses and displacements")

        H, xedges, yedges = np.histogram2d(self.center(displacements[:,0]), self.center(displacements[:,1]), bins = 200, range = [(-10,10),(-10,10)])

        if verbose: print("Made histogram")

        fig = plt.figure()
        plt.pcolor(xedges, yedges, H, norm=LogNorm(vmin=H.min(), vmax=H.max()), cmap='gist_yarg_r')
        if plotting: plt.colorbar()

        if verbose: print("Done\n")

        return xedges, yedges, H

    def density_1d(self, plotting = True, fitting_func = 'hern', verbose = True):
        """
        TODO: Documentation
        """

        if fitting_func == 'hern':
            fitting_func = lambda r, a: sum(self.masses)/(2*pi) * (a / r) * 1/(r + a)**3

        if verbose: print("Mass density plot")
            
        self.masses = self.get_values("Masses")
        self.displacements = self.center(self.get_values("Coordinates"))

        if verbose: print("Made masses and displacements")

        self.distances = self.make_magnitudes(self.displacements)

        if verbose: print("Made distances")
        
        density_hist = self.make_densities(bins = np.logspace(np.log10(2*10**-2), np.log10(50), 51))

        if verbose: print("Made density histogram")

        if fitting_func is not None: 

            best_param = curve_fit(fitting_func, density_hist[0], density_hist[1])[0]
            best_fit_func = lambda r: fitting_func(r, best_param)

            fit_x = density_hist[0]
            fit_y = np.array([best_fit_func(r) for r in density_hist[0]])

            if verbose: print("Made best fit plot")

            plt.plot(fit_x, fit_y)
              
        plt.plot(density_hist[0],density_hist[1])
        plt.xscale('log')
        plt.yscale('log')

        if plotting: plt.show()

        if verbose: print("Done\n")

        return density_hist

    def cheap_density(self, verbose = True):
        """
        TODO: Documentation
        """

        file_exists = os.path.exists(self.output_file + '/density/' + self.timeslice)

        if file_exists and self.overwriting or not file_exists:
            if verbose: print("Mass density plot")

            self.masses = self.get_values("Masses")    
            self.displacements = self.center(self.get_values("Coordinates"))

            if verbose: print("Made masses and displacements")

            self.distances = self.make_magnitudes(self.displacements)

            if verbose: print("Made distances")

            sample_bins = np.logspace(np.log10(2*10**-2), np.log10(50), 51)
            density_hist = self.get_density(sample_bins)

            if verbose: print("Made density histogram")
            
            data_destination = self.output_file + '/density/' + self.timeslice + '.txt'
            np.savetxt(data_destination, np.array([sample_bins[:-1], density_hist]))

            if verbose: print("Written to " + data_destination + "\n")

            return sample_bins[:-1], density_hist

    def velocity(self, plotting = True, v0 = True, beta = True, verbose = True):
        """
        TODO: Documentation
        """

        if verbose: print("Speed plot")

        self.masses = self.get_values("Masses")

        if verbose: print("Made masses")
        
        self.displacements = self.center(self.get_values("Coordinates"))

        if verbose: print("Made displacements")

        self.distances = self.make_magnitudes(self.displacements)

        if verbose: print("Made distances")

        self.velocities = self.get_values("Velocities")

        if verbose: print("Made velocities")

        self.speeds = self.make_magnitudes(self.velocities)
        
        if verbose: print("Made speeds")

        sample_bins = np.logspace(np.log10(2*10**-2),np.log10(50),51)
        v0, betas = self.get_v0_beta(sample_bins, verbose)
        
        print("Made velocity dispersions and beta anisotropy")

        speed_hist = self.make_hist(self.speeds, bins = sqrt(len(speeds)))

        print("Made histogram")

        if v0:
            plt.figure()
            plt.plot(sample_bins[:-1], v0)
            plt.title("Velocity Dispersions")
            plt.xlabel("$r$ (kpc)")
            plt.ylabel("v0 (km/s)")
            plt.xscale("log")

        if beta:
            plt.figure()
            plt.plot(sample_bins[:-1], betas)
            plt.title("Beta Anisotropy")
            plt.xlabel("$r$ (kpc)")
            plt.ylabel("Beta")
            plt.xscale("log")
        
        plt.figure()
        plt.plot(speed_hist[0], speed_hist[1])
        plt.title("Velocity Distribution")
        plt.xlabel("$v$ (km/s)")
        plt.ylabel("Frequency")

        if plotting: plt.show()

        print("Done\n")

        return speed_hist, v0, betas

    def excited_ratio(self, verbose = True):
        """
        TODO: Documentation
        """
        ratio = np.mean(self.get_values("SIDM_State"))
                        
        data_destination = self.output_file + '/ratios.txt'
        
        prev_ratios = np.loadtxt(data_destination, ndmin = 1)
            
        prev_ratios[int(self.timeslice)] = ratio
        np.savetxt(data_destination, prev_ratios)

        if verbose: print("Written to " + data_destination + "\n")
        
        return ratio
