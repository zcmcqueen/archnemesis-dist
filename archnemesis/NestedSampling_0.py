from archnemesis import *
import scipy
import pymultinest
from pymultinest.solve import solve
import os
import corner

class NestedSampling_0:
    
    def __init__(self, N_LIVE_POINTS=400):
        
        self.N_LIVE_POINTS = N_LIVE_POINTS
            
    def reduced_chi_squared(self, a,b,err):
        return np.sum(((a.flatten() - b.flatten())**2)/(len(a.flatten())*(err.flatten()**2)))

    def LogLikelihood(self,cube):
        
        self.ForwardModel.Variables.XN[self.vars_to_vary] = cube 
        
        original_stdout = sys.stdout  
        try:
            sys.stdout = open(os.devnull, 'w')  # Redirect stdout
            YN = self.ForwardModel.nemesisfm()
        finally:
            sys.stdout.close()  # Close the devnull
            sys.stdout = original_stdout  # Restore the original stdout
        
        return -np.log(self.reduced_chi_squared(self.Y,YN,self.Y_ERR))
    
    def Prior(self, cube):
        
        cube1 = cube.copy()
        for i in range(len(self.vars_to_vary)):
              cube1[i] = self.priors[i](cube1[i])

        return cube1
    
    def make_plots(self):
        a = pymultinest.Analyzer(n_params = len(self.parameters), outputfiles_basename = self.prefix)
        s = a.get_stats()

        print('Creating marginal plot ...')
        data = a.get_data()[:,2:]
        weights = a.get_data()[:,0]

        mask = weights > 1e-4

        corner.corner(data[mask,:], weights=weights[mask], 
            labels=self.parameters, show_titles=True)
        plt.savefig(self.prefix + 'corner.png')
        plt.close()

def coreretNS(runname,Variables,Measurement,Atmosphere,Spectroscopy,Scatter,Stellar,Surface,CIA,Layer):

    from archnemesis import ForwardModel_0
    from archnemesis import NestedSampling_0
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    NestedSampling = NestedSampling_0()
    
    NestedSampling.ForwardModel = ForwardModel_0(runname=runname, Atmosphere=Atmosphere,Surface=Surface,
                                  Measurement=Measurement,Spectroscopy=Spectroscopy,
                                  Stellar=Stellar,Scatter=Scatter,CIA=CIA,Layer=Layer,Variables=Variables)

    NestedSampling.XA = Variables.XA
    NestedSampling.XA_ERR = np.sqrt(Variables.SA.diagonal())
    NestedSampling.Y = Measurement.Y
    NestedSampling.Y_ERR = np.sqrt(Measurement.SE.diagonal())

    NestedSampling.vars_to_vary = [i for i in range(len(NestedSampling.XA)) if NestedSampling.XA_ERR[i]>1e-7]

    NestedSampling.priors = []

    for i in NestedSampling.vars_to_vary:
        dist_code = 1                              ### PLACEHOLDER - need to add custom distributions!
        if dist_code == 0:
            NestedSampling.priors.append(norm(NestedSampling.XA[i], NestedSampling.XA_ERR[i]).ppf)
        elif dist_code == 1:
            NestedSampling.priors.append(lambda x, i=i: x * (NestedSampling.XA[i] + NestedSampling.XA_ERR[i] - \
                                                             NestedSampling.XA[i] + NestedSampling.XA_ERR[i]) + \
                                                             NestedSampling.XA[i] - NestedSampling.XA_ERR[i])
        else:  
            print('DISTRIBUTION ID NOT DEFINED!', flush = True)

    # name of the output files
    NestedSampling.prefix = "chains/"
    if rank == 0:
        if not os.path.exists(NestedSampling.prefix):
            os.makedirs(NestedSampling.prefix)
    comm.barrier()
    
    NestedSampling.parameters = [str(i) for i in NestedSampling.vars_to_vary]
    # run MultiNest
    NestedSampling.result = solve(LogLikelihood=NestedSampling.LogLikelihood, 
                                  Prior=NestedSampling.Prior, 
                                  n_dims=len(NestedSampling.parameters), 
                                  outputfiles_basename=NestedSampling.prefix, 
                                  verbose=True, 
                                  n_live_points = NestedSampling.N_LIVE_POINTS)

    
    if rank == 0:
        print()
        print('Evidence: %(logZ).1f +- %(logZerr).1f' % NestedSampling.result)
        print()
        print('Parameter values:')
        for name, col in zip(NestedSampling.parameters, NestedSampling.result['samples'].transpose()):
            print('%15s : %.3f +- %.3f' % (name, col.mean(), col.std()))
    comm.barrier()
    return NestedSampling