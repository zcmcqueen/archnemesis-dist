from archnemesis import *
import scipy
import pymultinest
from pymultinest.solve import solve
import os
import corner

class NestedSampling_0:
    
    def __init__(self, N_LIVE_POINTS=400):
        
        """
        Inputs
        ------
        @param N_LIVE_POINTS: int,
            Number of live points in retrieval 

        Methods
        -------
        NestedSampling.reduced_chi_squared()
        NestedSampling.LogLikelihood()
        NestedSampling.Prior()
        NestedSampling.make_plots()
        """
        
        
        self.N_LIVE_POINTS = N_LIVE_POINTS
            
    def reduced_chi_squared(self, a,b,err):
        """
        Calculate chi^2/n statistic.
        """        
        
        return np.sum(((a.flatten() - b.flatten())**2)/(len(a.flatten())*(err.flatten()**2)))

    def LogLikelihood(self,cube):
        """
        Compute likelihood - run a forward model and compare to spectrum.
        """   
        
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
        """
        Map unit cube to prior distributions.
        """  
        
        cube1 = cube.copy()
        for i in range(len(self.vars_to_vary)):
              cube1[i] = self.priors[i](cube1[i])

        return cube1
    
    def make_plots(self):
        """
        Cornerplot of results.
        """ 
        
        prior_means = self.XA
        prior_stds = self.XA_ERR

        # Initialize the analyzer
        a = pymultinest.Analyzer(n_params=len(self.parameters), outputfiles_basename=self.prefix)
        s = a.get_stats()

        print('Creating marginal plot ...')

        # Extract data and weights
        data_array = a.get_data()
        weights = data_array[:, 0]
        data = data_array[:, 2:]

        # Apply weight mask (optional, depending on your data)
        mask = weights > 1e-4
        data_masked = data[mask, :]
        weights_masked = weights[mask]

        # Generate prior samples from Gaussian distributions
        num_prior_samples = 1000000
        prior_samples = np.zeros((num_prior_samples, len(self.parameters)))

        for i in range(len(self.parameters)):
            prior_samples[:, i] = np.random.normal(
                loc=prior_means[i],
                scale=prior_stds[i],
                size=num_prior_samples
            )

        # Combine prior and posterior samples for consistent axis ranges
        combined_samples = np.vstack((data_masked, prior_samples))

        # Determine axis ranges from combined samples
        ranges = []
        for i in range(len(self.parameters)):
            min_val = np.min(data_masked[:, i])
            max_val = np.max(data_masked[:, i])
            ranges.append((min_val, max_val))

            
        # Plot the corner plot for posterior samples
        
        figure = corner.corner(
            prior_samples,
            labels=self.parameters,
            color='red',
            range=ranges,
            bins=50,  # Match bins with the posterior histogram
            hist_kwargs={'density': True},
            plot_contours=True,
            fill_contours=False,
            contour_colors=['red'],
            plot_datapoints=False,  # Adjust transparency
            zorder = -1,
            smooth = 1.0# Plot on the same figure
        )
        
        figure = corner.corner(
            data_masked,
            weights=weights_masked,
            labels=self.parameters,
            show_titles=True,
            color='blue',
            range=ranges,
            bins=50,  # Adjust as needed
            hist_kwargs={'density': True},
            plot_contours=True,
            fill_contours=False,
            contour_colors=['blue'],
            fig=figure ,
            zorder = 1,
            smooth = 1.0,
            data_kwargs={'alpha': 0.5},  # Adjust transparency
        )


            

        # Adjust the diagonal plots (1D histograms) to add legends
        axes = np.array(figure.axes).reshape((len(self.parameters), len(self.parameters)))

        for i in range(len(self.parameters)):
            ax = axes[i, i]
            # Add legends to diagonal plots
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(handles, labels)
            
        from matplotlib.lines import Line2D

        legend_elements = [
            Line2D([0], [0], color='blue', lw=2, label='Posterior'),
            Line2D([0], [0], color='red', lw=2, label='Prior')
        ]

        figure.legend(handles=legend_elements, loc='upper right')
        
        plt.savefig(self.prefix + 'corner.png')
        plt.close()

def coreretNS(runname,Variables,Measurement,Atmosphere,Spectroscopy,Scatter,Stellar,Surface,CIA,Layer):
    """
        FUNCTION NAME : coreretNS()
        
        DESCRIPTION : 

            This subroutine runs Nested Sampling to fit an atmospheric model to a spectrum, and gives
            a good idea of the distribution of fitted parameters.

        INPUTS :
       
            runname :: Name of the Nemesis run
            Variables :: Python class defining the parameterisations and state vector
            Measurement :: Python class defining the measurements 
            Atmosphere :: Python class defining the reference atmosphere
            Spectroscopy :: Python class defining the spectroscopic parameters of gaseous species
            Scatter :: Python class defining the parameters required for scattering calculations
            Stellar :: Python class defining the stellar spectrum
            Surface :: Python class defining the surface
            CIA :: Python class defining the Collision-Induced-Absorption cross-sections
            Layer :: Python class defining the layering scheme to be applied in the calculations

        OUTPUTS :

            NestedSampling :: Python class containing information from the retrieval.
 
        CALLING SEQUENCE:
        
            NestedSampling = coreretNS(runname,Variables,Measurement,Atmosphere,Spectroscopy,Scatter,Stellar,Surface,CIA,Layer)
 
        MODIFICATION HISTORY : Joe Penn (09/10/24)

    """
    
    
    from archnemesis import ForwardModel_0
    from archnemesis import NestedSampling_0
    from mpi4py import MPI

    # This function should be launched in parallel. We set up the MPI environment.
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Defining the NestedSampling class
    NestedSampling = NestedSampling_0()
    
    NestedSampling.ForwardModel = ForwardModel_0(runname=runname, Atmosphere=Atmosphere,Surface=Surface,
                                  Measurement=Measurement,Spectroscopy=Spectroscopy,
                                  Stellar=Stellar,Scatter=Scatter,CIA=CIA,Layer=Layer,Variables=Variables)

    NestedSampling.XA = Variables.XA
    NestedSampling.XA_ERR = np.sqrt(Variables.SA.diagonal())
    NestedSampling.Y = Measurement.Y
    NestedSampling.Y_ERR = np.sqrt(Measurement.SE.diagonal())

    # Setting up prior distributions - right now, this just sets up log-gaussians with a standard deviation
    # equal to what is specified in the .apr file. There is support for log-uniform distributions for dist_code=1.
    
    NestedSampling.vars_to_vary = [i for i in range(len(NestedSampling.XA)) if NestedSampling.XA_ERR[i]>1e-7]

    NestedSampling.priors = []

    for i in NestedSampling.vars_to_vary:
        dist_code = 0                              ### PLACEHOLDER - need to add custom distributions!
        if dist_code == 0:
            NestedSampling.priors.append(scipy.stats.norm(NestedSampling.XA[i], NestedSampling.XA_ERR[i]).ppf)
        elif dist_code == 1:
            NestedSampling.priors.append(lambda x, i=i: x * (NestedSampling.XA[i] + NestedSampling.XA_ERR[i] - \
                                                             NestedSampling.XA[i] + NestedSampling.XA_ERR[i]) + \
                                                             NestedSampling.XA[i] - NestedSampling.XA_ERR[i])
        else:  
            print('DISTRIBUTION ID NOT DEFINED!', flush = True)

    # Making the retrieval folder
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

    #Print parameters
    if rank == 0:
        print()
        print('Evidence: %(logZ).1f +- %(logZerr).1f' % NestedSampling.result)
        print()
        print('Parameter values:')
        for name, col in zip(NestedSampling.parameters, NestedSampling.result['samples'].transpose()):
            print('%15s : %.3f +- %.3f' % (name, col.mean(), col.std()))
    comm.barrier()
    return NestedSampling