from dm_plotter import *

p = Plotter(test_dir = '../snapshots/mnt/',
            tests = ['Test_' + str(n + 1) for n in range(10)],
            timeslices = ['0' + ('0' if n < 10 else '') + str(n) for n in range(21)],
            verbose = True,
            overwriting = False)

p.plot_density()
p.plot_sup_factors()
p.plot_ratios()
