import os
import numpy as np
from dm_analysis import *
from scipy import integrate

class Plotter():
    """ Utility class for plotting data analyzed by Analyzer. """
    
    def __init__(self, test_dir, tests = 'all', timeslices = 'auto', overwriting = False, verbose = False):
        """
        TODO: Documentation
        """
        init_dir = os.path.abspath(os.getcwd())

        os.chdir(test_dir)
        if tests == 'all':
            self.tests = [f if f.startswith('Test') else '' for f in os.listdir()]
            self.tests[:] = [test for test in self.tests if test != '']
            self.tests.sort()
        else:
            self.tests = tests

        if timeslices == 'auto':
            # Get timeslices in the format '###'.
            os.chdir(self.tests[0] + '/output')
            self.timeslices = [f.split('_')[1].split('.')[0] if f.startswith('snap_') else '' for f in os.listdir()]
            self.timeslices[:] = [t for t in self.timeslices if t != '']
            self.timeslices.sort()

            os.chdir('../..')
        else:
            self.timeslices = timeslices

        # Make the analyzer file system for output plots and data.
        file_system = ['analysis', 'analysis/plots',
                       'analysis/plots/density', 'analysis/plots/gifs',
                       'analysis/plots/ratios', 'analysis/plots/sup_factors'] \
                    + ['analysis/' + test for test in self.tests] \
                    + ['analysis/' + test + '/density' for test in self.tests]
        
        for f in file_system:
            if not os.path.exists(f): os.mkdir(f)
        
        self.overwriting = overwriting
        self.verbose = verbose

        # Functions in the analyzer that get the appropriate type of data.
        self.data_func = {'density' : 'cheap_density',
                          'ratios' : 'excited_ratio'}

        # Directory for the plots/analyzer output.
        self.a_dir = test_dir + 'analysis/'
        self.test_dir = test_dir

        os.chdir(init_dir)

    def load_time_file(self, data, test, t = None):
        """
        TODO: Documentation
        """
        # Two cases, the plot is over a specific snapshot (t != None), or it is over the time domain (t = None).
        if t is not None:
            
            try:
                if self.overwriting:
                    # If we're overwriting, just break to the exception without trying to read the file.
                    raise(FileNotFoundError)
                
                else:
                    # Load the file if it's already made.
                    info = np.loadtxt(self.a_dir + test + '/' + data + '/' + t + '.txt', ndmin = 1)
                    if self.verbose: print("Found " + data + " for " + test + '/' + data + '/' + t + '.txt')
                                           
                return info
            
            except FileNotFoundError:
                # If we're overwriting or the file isn't defined, we make it.
                if self.verbose: print("Didn't find " + data + " for " + test + '/' + data + '/' + t + '.txt' + "\n" + "Making now...")

                analyzer = Analyzer(self.test_dir, test, t)
                info = eval('analyzer.' + self.data_func[data] + '(verbose = False)')
                
                return info
            
        else:
            try:
                if self.overwriting:
                    # If we're overwriting, just break to the exception without trying to read the file.
                    raise(FileNotFoundError)

                else:
                    # Load the file if it's already made.
                    info = np.loadtxt(self.a_dir + test + '/' + data + '.txt', ndmin = 1)
                    if self.verbose: print("Found " + data + " for " + test + '/' + data + '.txt')

            except FileNotFoundError:
                # If we're overwriting or the file isn't defined, we make it.
                # Initialize the values to an invalid number, -1.
                if self.verbose: print("Didn't find " + data + " for " + test + '/' + data + '.txt' + "\n" + "Making now...")
                info = np.repeat(-1, len(self.timeslices))
                np.savetxt(self.a_dir + test + '/' + data + '.txt', info)

            for n, t in enumerate(self.timeslices):
                # Generate new values in invalid spots.
                if self.verbose: print("Didn't find " + data + " for " + test + '/' + data + '/' + t + '.txt' + "\n" + "Making now...")

                if info[n] == -1:
                    analyzer = Analyzer(self.test_dir, test, t)
                    info[n] = eval('analyzer.' + self.data_func[data] + '(verbose = False)')

            return info

    def make_label(self, test, i_or_ie = False):
        """
        TODO: Documentation
        """
        label = str(int(test.split('_')[1])/10)
        
        if i_or_ie:
            label.append(' ' + test.split('_')[2].capitalize())
            
        return label
                            
    def plot_density(self):
        """
        TODO: Documentation
        """
        for t in self.timeslices:
            # For all of the time slices...
            fig = plt.figure(t)
            ax = fig.add_subplot(111)
            
            for test in self.tests:
                # For each test, load up the density distribution.
                hist = self.load_time_file('density', test, t)

                ax.plot(hist[0], hist[1], label = self.make_label(test))

            hist = self.load_time_file('density', self.tests[0], '000')
            ax.plot(hist[0], hist[1], '--', color = '0.5', label = 'hernquist')

            # Formatting the plot.
            plt.title("Mass Density - Snapshot " + t)
            plt.ylim(10**-9,10**0)
            plt.legend(loc = 'upper right')
            plt.xlabel("$r$ (kpc)")
            plt.ylabel("Density ($10{^10}$ Solar Masses/kpc)")
            plt.xscale('log')
            plt.yscale('log')

            # Saving the plot to our density plot file.
            plt.savefig(self.a_dir + 'plots/density/' + t + '.png')
            plt.close()

    def plot_sup_factors(self):
        """
        TODO: Documentation
        """
        # We defined the suppression factor to be the integral over the log-space in both the time and density dimensions.

        total_fig = plt.figure()
        total_ax = total_fig.add_subplot(111)
        
        for test in self.tests:
            # For each test, load up the log-space of its initial distribution.
            initial_dist = np.log10(self.load_time_file('density', test, '000'))
            
            sup_factors = []

            fig = plt.figure()
            ax = fig.add_subplot(111)
            
            for t in self.timeslices:
                # For each time slice, load the log-space of its distribution.
                t_dist = np.log10(self.load_time_file('density', test, t))
                
                # Now we create a new plot along the time slices, that represents the difference
                # in the log-space between the initial and particular time slice distribution.
                diff_dist = np.array([initial_dist[0], t_dist[1] - initial_dist[1]])

                # Set missing data to zero so we don't break the integral.
                diff_dist[1][np.array([np.isnan(n) for n in diff_dist[1]])] = 0

                # Integrate over the time axis to get our suppression factor.
                sup_factors.append(abs(integrate.trapz(diff_dist[1], diff_dist[0])))

            ax.plot(range(len(self.timeslices)), sup_factors)
            total_ax.plot(range(len(self.timeslices)), sup_factors, label = self.make_label(test))

            # Formatting the plot.
            plt.title("Suppression Factor - " + ' '.join(test.split('_')))
            plt.xlabel("$t$ (Gyr)")
            plt.ylabel("Factor")
            
            plt.savefig(self.a_dir + 'plots/sup_factors/' + test + '.png')
            plt.close()

        plt.figure(total_fig.number)

        plt.title("Suppression Factor - All")
        plt.legend(loc = 'upper left')
        plt.xlabel("$t$ (Gyr)")
        plt.ylabel("Factor")

        plt.savefig(self.a_dir + 'plots/sup_factors/all.png')
        plt.close()

    def plot_ratios(self):
        """
        TODO: Documentation
        """
        total_fig = plt.figure()
        total_ax = total_fig.add_subplot(111)

        # Initialize the plot for each group of tests.
        fig = plt.figure()
        ax = fig.add_subplot(111)
            
        for test in self.tests:
            # For each test, load up the ratios over the time slices.
            ratios = self.load_time_file('ratios', test)
 
            # Plot the test.
            ax.plot(range(len(self.timeslices)), ratios, label = self.make_label(test))
            total_ax.plot(range(len(self.timeslices)), ratios, label = self.make_label(test))

            # Formatting the plot.
            plt.title("Excited Ratio - " + self.make_label(test))
            plt.legend(loc = 'upper right')
            plt.xlabel("$t$ (Gyr)")
            plt.ylabel("Ratio (Excited/Ground)")

            plt.savefig(self.a_dir + 'plots/ratios/' + str(int(100*float(self.make_label(test)))) + '.png')
            plt.close()

        # Make a final plot with all the ratios on it.
        plt.figure(total_fig.number)

        plt.title("Excited Ratio - All")
        plt.legend(loc = 'center right')
        plt.xlabel("$t$ (Gyr)")
        plt.ylabel("Ratio (Excited/Ground)")

        plt.savefig(self.a_dir + 'plots/ratios/all.png')
        plt.close()
