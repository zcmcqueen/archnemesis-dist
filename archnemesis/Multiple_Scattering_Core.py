import os
import numpy as np
from numba import njit,prange

# @njit(fastmath=True, error_model='numpy')
# def phase1(calpha, iscat, cons, icons=0, icont=None, ncont=None, vwave=None, pfunc = None, xmu = None):
#     pi = np.pi
#     calpha = min(max(calpha, -1.0), 1.0)
#     if iscat == 0:
#         p = 0.75 * (1.0 + calpha * calpha)
#     elif iscat == 1:
#         p = 1.0
#     elif iscat == 2:
#         f1 = cons[0]
#         f2 = 1.0 - f1
#         hg11 = 1.0 - cons[1] * cons[1]
#         hg12 = 2.0 - hg11
#         hg21 = 1.0 - cons[2] * cons[2]
#         hg22 = 2.0 - hg21
#         p = f1 * hg11 / np.sqrt(hg12 - 2.0 * cons[1] * calpha) ** 3 + f2 * hg21 / np.sqrt(hg22 - 2.0 * cons[2] * calpha) ** 3
#     elif iscat == 3:
#         cons[0] = 0.25 / pi
#         p = 0.0
#         xf = calpha * calpha
#         for k in range(1, (icons + 1) // 2 + 1):
#             n0 = 2 * k - 1
#             n1 = 2 * k
#             x0 = 1.0
#             x1 = calpha
#             pa0 = 0.0
#             pa1 = 0.0
#             for i in range(1, k + 1):
#                 pa0 += x0 * jcoef[i - 1, n0 - 1] / jdiv[k - 1]
#                 pa1 += x1 * jcoef[i - 1, n1 - 1] / jdiv[k - 1]
#                 x0 *= xf
#                 x1 *= xf
#             p += cons[n0] * pa0 + cons[n1] * pa1
#     elif iscat == 4:
#         p = np.interp(calpha,xmu,pfunc)
        
#     else:
#         print('Error invalid scattering option.')
#         return None
#     p /= 4.0 * pi
#     return p

@njit
def phasint2(nphi, ic, nmu, mu, iscat, cons, ncons, icont, ncont, pfunc, xmu, idump=0):
    
    pi = np.pi
    dphi = 2.0 * pi / nphi

    phi = np.arange(nphi + 1) * dphi
    sth = np.sqrt(1.0 - mu**2)
    mu2d = mu[:, None]
    
    cos_ic_phi = np.cos(ic * phi)
    cos_phi_grid = np.cos(phi)

    # Broadcasting
    sth_sth = sth[:, None] * sth[None, :]
    mu_mu = mu2d * mu2d.T
    
    cpl = sth_sth[:, :, None] * cos_phi_grid + mu_mu[:, :, None]
    cmi = sth_sth[:, :, None] * cos_phi_grid - mu_mu[:, :, None]
    #Cutting out phase1 for now

    if iscat == 0:
        pl = 0.75 * (1.0 + cpl * cpl)/ (4 * np.pi)
        pm = 0.75 * (1.0 + cmi * cmi)/ (4 * np.pi)
    else:
        pl = np.interp(cpl, xmu, pfunc) #phase1(cpl, iscat, cons, ncons, icont, ncont, vwave, pfunc, xmu)
        pm = np.interp(cmi, xmu, pfunc) #phase1(cmi, iscat, cons, ncons, icont, ncont, vwave, pfunc, xmu)        
        
    wphi = np.full(phi.shape, dphi)
    wphi[0]  = 0.5 * dphi
    wphi[-1] = 0.5 * dphi
    
    if ic == 0:
        wphi /= (2.0 * pi)
    else:
        wphi /= pi
    
    wphi = wphi[None, None, :]
    
    plx = pl * cos_ic_phi
    pmx = pm * cos_ic_phi
    
    pplpl = np.sum(wphi * plx, axis=2)
    pplmi = np.sum(wphi * pmx, axis=2)

    return (pplpl, pplmi)


@njit(fastmath=True, error_model='numpy')
def hansen(ic, ppl, pmi, wtmu, nmu, fc):
    '''
    Subroutine to normalise the phase function using the method described in Hansen (1971,J.ATM.SCI., V28, 1400)

    PPL,PMI are the forward and backward parts of the azimuthally-integrated
    phase function. The normalization of the true phase fcn is:
    integral over sphere [ P(mu,mu',phi) * dO] = 1
    where dO os the element of solid angle and phi is the azimuthal angle.
    '''
    
    pi = np.pi
    x1 = 2.0 * pi
    if ic == 0:
        rsum = np.zeros(nmu, dtype=float)
        for j in range(nmu):
            rsum[j] = np.sum(pmi[:, j] * wtmu) * x1
        niter = 0
        tsum = np.zeros(nmu)
        
        for niter in range(10000):
            for j in range(nmu):
                tsum[j] = np.sum(ppl[:, j] * wtmu * fc[:, j]) * x1
            testj = np.abs(rsum + tsum - 1.0)
            test = np.max(testj)
            if test < 1e-14:
                break
            for j in range(nmu):
                xj = (1.0 - rsum[j]) / tsum[j]
                for i in range(j+1):
                    xi = (1.0 - rsum[i]) / tsum[i]
                    fc[i, j] = 0.5 * (fc[i, j] * xj + fc[j, i] * xi)
                    fc[j, i] = fc[i, j]
    ppl *= fc
    return (ppl, fc)

@njit(fastmath=True)
def calc_pmat6(ic, mu, wtmu, nmu, iscat, cons8, ncons, norm, icont, ncont, nphi, fc, pfunc, xmu):
    pplpl, pplmi = phasint2(nphi, ic, nmu, mu, iscat, cons8, ncons, icont, ncont, pfunc, xmu)

    if norm == 1:
        pplpl, fc = hansen(ic, pplpl, pplmi, wtmu, nmu, fc)

    return (pplpl, pplmi, fc)
            
@njit(fastmath=True)
def matmul(A, B):
    '''
    Matrix multiplication
    '''
    # Get the dimensions of the matrices
    rows_A, cols_A = A.shape
    rows_B, cols_B = B.shape

    # Ensure the matrices are compatible for multiplication
    if cols_A != rows_B:
        raise ValueError("Number of columns in A must be equal to number of rows in B")

    # Initialize the result matrix with zeros
    result = np.zeros((rows_A, cols_B))

    # Perform matrix multiplication using nested loops
    for i in range(rows_A):
        for j in range(cols_B):
            for k in range(cols_A):
                result[i, j] += A[i, k] * B[k, j]

    return result

@njit
def frob(r):
    return np.sqrt((r*r).sum())

@njit(fastmath=True)
def add(r1, t1, j1, e, nmu,ic):

    bcom = matmul(r1,r1)
    acom = np.linalg.inv(e-bcom)
    #if frob(r1)>0.1: # Frobenius norm < 0.1 for approximation to keep error under 1e-4
    #    acom = np.linalg.solve(e - rsq, e)
    #else: 
    #    acom = e + rsq
    ccom = matmul(t1,acom)
    rans = matmul(ccom,r1)
    acom = matmul(rans,t1)
    rans = r1 + acom
    tans = matmul(ccom,t1)

    if ic==0:
        jcom = matmul(r1,j1)
        jcom = jcom + j1
        jans = matmul(ccom,jcom)
        jans = jans + j1
    else:
        jans = j1

    return rans, tans, jans


@njit(fastmath=True)
def numba_diagonal(arr):
    rows, cols = arr.shape
    length = min(rows, cols)
    diagonal = np.empty(length, arr.dtype)
    for i in range(length):
        diagonal[i] = arr[i, i]
    return diagonal

@njit(fastmath=True)
def numba_fill_diagonal(arr, vec):
    for i in range(len(vec)):
        arr[i, i] = vec[i]
        
@njit(fastmath = True)
def numba_sum_diagonal(arr):
    sum_ = 0
    for i in range(arr.shape[0]):
        sum_ += arr[i,i]
    return sum_

@njit(fastmath = True, error_model='numpy')
def double1(ic,nmu,cc,pplpl,pplmi,omega,taut,bc,mminv,e,raman = False): 
    
    ipow0 = 12

    con = omega * np.pi
    del01 = 0.0
    if ic == 0:
        del01 = 1.0
    
    con *= (1.0 + del01)
    
    #Computation of Gamma++ (Plass et al. 1973)
    #GPLPL = MMINV*(E - CON*PPLPL*CC)
    acom = matmul(pplpl,cc)*con
    bcom = e - acom
    gplpl = matmul(mminv,bcom)
    
    #Computation of Gamma+- (Plass et al. 1973)
    #GPLMI = MMINV*CON*PPLMI*CC
    acom = matmul(pplmi,cc)*con
    gplmi = matmul(mminv,acom)
    
    nn = int(np.log2(taut) + ipow0)
    xfac = 1.0 / (2.0 ** nn) if nn >= 1 else 1.0
    tau0 = taut * xfac
    
    # Computation of R, T and J for initial layer
    t1 = e - tau0 * gplpl.transpose()
    r1 = tau0 * gplmi.transpose()
    
    j1 = np.zeros((nmu,1))
    if ic == 0:
        for i in range(nmu):
            j1[i] = (1.0 - omega) * bc * tau0 * mminv[i,i]

    if nn < 1:
        return r1, t1, j1
    
    for n in range(nn):
        r1, t1, j1 = add(r1, t1, j1, e, nmu, ic)
   
    return r1, t1, j1

@njit(fastmath = True)
def addp(r1, t1, j1, iscat1,e, rsub, tsub, jsub, nmu): 

    if iscat1 == 1:
        rsq = np.zeros_like(r1)
        bcom = np.zeros_like(r1)
        ccom = np.zeros_like(r1)
        rans = np.zeros_like(r1)
        tans = np.zeros_like(r1)
        jcom = np.zeros_like(j1)
        jans = np.zeros_like(j1)
        
        
        # Second layer is scattering
        rsq = matmul(rsub,r1)
        acom = np.linalg.inv(e-rsq)
        ccom = matmul(t1,acom)
#        if frob(rsq)>0.01:
#            acom = np.linalg.solve(e - rsq, e)
#            ccom = matmul(t1,acom)
#        else:
#            acom = e+rsq
#            ccom = matmul(t1,acom)
        rans = matmul(ccom,rsub)
        bcom = matmul(rans,t1)
        rans = r1 + bcom
        tans = matmul(ccom,tsub)
        jcom = matmul(rsub,j1)
        jcom += jsub
        jans = matmul(ccom,jcom)
        jans += j1
    else:
        # Second layer is non-scattering
        jcom = np.zeros_like(j1)
        tans = np.zeros((nmu,nmu))
        rans = np.zeros((nmu,nmu))
        jans = np.zeros((nmu,1))
        
        jcom = matmul(rsub,j1)
        jcom += jsub
        
        for i in range(nmu):
            ta = t1[i, i]
            for j in range(nmu):
                tb = t1[j, j]   
                tans[i, j] = tsub[i, j] * ta
                rans[i, j] = rsub[i, j] * ta * tb
            jans[i, 0] = j1[i, 0] + ta * jcom[i, 0]
            
    return rans, tans, jans

@njit(fastmath = True)
def angle_quadrature(solar,sol_ang,emiss_ang,mu,nmu):

    if sol_ang > 90.0:
        zmu0 = np.cos(np.radians(180 - sol_ang))
        solar1 = solar*0.0
    else:
        zmu0 = np.cos(np.radians(sol_ang))
        solar1 = solar

    zmu = np.cos(np.radians(emiss_ang))

    isol = 0
    for j in range(nmu-1):
        if zmu0 <= mu[j] and zmu0 > mu[j+1]:
            isol = j
    if zmu0 <= mu[nmu-1]:
        isol = nmu - 2

    iemm = 0
    for j in range(nmu-1):
        if zmu <= mu[j] and zmu > mu[j+1]:
            iemm = j
    if zmu <= mu[nmu-1]:
        iemm = nmu - 2

    u = (mu[isol] - zmu0) / (mu[isol] - mu[isol+1])
    t = (mu[iemm] - zmu) / (mu[iemm] - mu[iemm+1])
    
    return solar1, isol, iemm, t, u 

@njit(fastmath = True, error_model='numpy')
def calc_rtj_matrix(ic,mu,wtmu,bb,tautot,tauscat,tauray,frac,ppln,pmin,pplr,pmir,cc=None,mminv=None,e=None):
    '''
    Calculate the Reflection, Transmission and Source matrices for a given atmospheric layer
    
    Inputs
    ______
    
    mu(nmu) :: Quadrature angles
    wtmu(nmu) :: Weights of the quadrature angles
    bb :: Black-body radiation for the layer 
    tautot :: Total optical depth of the layer (absorption + scattering)
    omega :: Aerosol scattering optical depth of the layer
    tauray :: Rayleigh scattering optical depth of the layer
    frac(ndust) :: Fraction of aerosol scattering by each aerosol population
    ppln(ndust,nmu,nmu) :: Phase matrix in + direction of each aerosol population
    pmin(ndust,nmu,nmu) :: Phase matrix in - direction of each aerosol population
    pplr(nmu,nmu) :: Phase matrix in + direction for Rayleigh scattering
    pmir(nmu,nmu) :: Phase matrix in - direction for Rayleigh scattering
    
    Outputs
    _______
    
    rl(nmu,nmu) :: Reflection matrix
    tl(nmu,nmu) :: Transmission matrix
    jl(nmu,1) :: Source matrix
    iscl :: Flag indicating whether layer is scattering (1) or non-scattering (0)
    '''

    nmu = len(mu)       #Number of quadrature angles
    ncont = len(frac)   #Number of aerosol populations
    omega = (tauscat+tauray)/tautot  #Single scattering albedo of the layer
    
    #Initialising arrays
    rl = np.zeros((nmu,nmu))
    tl = np.zeros((nmu,nmu))
    jl = np.zeros((nmu,1))
    
    #Calculating matrices
    if cc is None:
        cc = np.zeros((nmu,nmu))
        numba_fill_diagonal(cc, wtmu)
    if e is None:
        e = np.identity(nmu)
    if mminv is None:
        mminv = np.zeros((nmu,nmu))
        numba_fill_diagonal(mminv,1./mu)
    
    #Calculating matrices
    if tautot == 0:     #If there is no atmospheric opacity
        iscl = 0
        rl[:,:] *= 0.0
        tl[:,:] *= 0.0
        jl[:,:] *= 0.0
        for i in range(nmu):
            tl[i,i] = 1.0

    elif omega == 0:  #If layer is not scattering
        iscl = 0
        rl[:,:] *= 0.0
        tl[:,:] *= 0.0
        for i1 in range(nmu):
            tex = -mminv[i1,i1]*tautot
            if tex > -200.0:
                tl[i1,i1] = np.exp(tex)
            else:
                tl[i1,i1] = 0.0

            jl[i1,0] = bb*(1.0 - tl[i1,i1])

    else:  #If layer is scattering
        pplpl = (tauray/(tauscat+tauray))*pplr[:,:]
        pplmi = (tauray/(tauscat+tauray))*pmir[:,:]

        for j1 in range(ncont):
            pplpl += tauscat/(tauscat+tauray)*ppln[j1,:,:]*frac[j1] 
            pplmi += tauscat/(tauscat+tauray)*pmin[j1,:,:]*frac[j1] 

        iscl = 1
        rl[:,:], tl[:,:], jl[:,:] = double1(ic,nmu,cc,pplpl,pplmi,omega,tautot,bb,mminv,e,raman = False)
    
    return rl,tl,jl,iscl



@njit(fastmath = True,parallel=False, cache = True, error_model='numpy')
def scloud11wave_core(phasarr, radg, sol_angs, emiss_angs, solar, aphis, lowbc, galb, mu1, wt1, nf,
                vwaves, bnu, taus, tauray,omegas_s, nphi,iray, lfrac):
    
    """
    Calculate spectrum using the adding-doubling method.

    Parameters
    ----------
    phasarr(NMODES,NWAVE,2,NPHAS):
        Contains phase functions (phasarr[:,:,0,:]) and corresponding angle grids (phasarr[:,:,1,:])
    radg(NWAVE,NMU): 
        Incident intensity at the bottom of the atm
    sol_ang(NPATH):
        Solar zenith angle (degrees)
    emiss_angs(NPATH):
        Emission zenith angle (degrees)
    solar(NWAVE):        
        Incident solar flux at the top of the atm
    aphi(NPATH):
        Azimuth angle between Sun and observer
    lowbc:
        Lower boundary condition: 0 = thermal (no surface), 1 = Lambert reflection (surface)
    galb(NWAVE):
        Ground albedo at the bottom
    mu1(NMU):  
        Zenith angle point quadrature
    wt1(NMU):  
        Zenith angle point quadrature
    nf:
        Maximum number of terms in azimuth Fourier expansion 
    vwaves(NWAVE):
        Input wavelengths.
    bnu(NWAVE,NLAY):
        Mean Planck function in each layer
    tau(NWAVE,NG,NLAY):
        Total optical thickness of each layer
    tauray(NWAVE,NLAY):
        Rayleigh optical thickness of each layer
    omegas(NWAVE,NG,NLAY):
        Single scattering albedo of each layer (aerosol + Rayleigh scattering)
    nphi:
        Number of azimuth integration ordinates
    iray:
        Flag for using rayleigh scattering
    lfrac(NWAVE,NCONT,NLAY):
        Fraction of scattering contributed by each type in each layer
    
    Returns
    -------
    rad(NWAVE):
        Upwards radiance at each wavelength
    """
    
    nwave = len(vwaves)
    nmu = len(mu1)
    ngeom = len(emiss_angs)
    ng = taus.shape[1]
    nlay = taus.shape[2]
    ncont = phasarr.shape[0]
    pi = np.pi

    ltot = nlay
    if lowbc > 0:
        ltot += 1

    xfac = 0.
    xfac = np.sum(mu1*wt1)
    xfac = 0.5/xfac    

    # Reset the order of angles
    mu = mu1[::-1]
    wtmu = wt1[::-1]
    
    # Setting up radiance matrices
    rad = np.zeros((ngeom,ng,nwave))

    yx = np.zeros((4))
    
    u0pl = np.zeros((nmu,1))
    utmi = np.zeros((nmu,1))
    umi = np.zeros((nlay,nmu))
    upl = np.zeros((nlay,nmu))

    ppln = np.zeros((ncont, nmu, nmu))
    pmin = np.zeros((ncont, nmu, nmu))
    pplr = np.zeros((nmu, nmu))
    pmir = np.zeros((nmu, nmu))
    
    rl = np.zeros((nmu,nmu))
    tl = np.zeros((nmu,nmu))
    jl = np.zeros((nmu,1))

    rbase = np.zeros((nmu,nmu))
    tbase = np.zeros((nmu,nmu))
    jbase = np.zeros((nmu,1))

    # Setting up matrices of constants 
    e = np.identity(nmu)
    mm = np.zeros((nmu, nmu))
    numba_fill_diagonal(mm, mu[:nmu])
    mminv = np.zeros((nmu, nmu))
    numba_fill_diagonal(mminv, 1./mu[:nmu])
    cc = np.zeros((nmu, nmu))
    numba_fill_diagonal(cc, wtmu)
    ccinv = np.zeros((nmu, nmu))
    numba_fill_diagonal(ccinv, 1/wtmu[:nmu])
    
    radg = radg[:,::-1]

    fc = np.ones((ncont+1,nmu,nmu))
    
    for ipath in range(ngeom):
        # Setting up path variables
        
        sol_ang = sol_angs[ipath]
        emiss_ang = emiss_angs[ipath]
        aphi = aphis[ipath]
        #nf = int(emiss_ang//3 + 1)
        
        if emiss_ang < 90.:
            lookdown = True    #Nadir-viewing geometry
            new_emi = emiss_ang
        elif emiss_ang > 90.:
            lookdown = False   #Upward-looking geometry
            new_emi = 180. - emiss_ang
            
        solar1, isol, iemm, t, u = angle_quadrature(solar,sol_ang,new_emi,mu,nmu)
        
        for ig in range(ng):
            # Getting correct layer properties for this g-ordinate
            
            tau = taus[:,ig,:]
            omegas = omegas_s[:,ig,:]
            
            for widx in range(nwave):
                converged = False
                conv1 = False
                defconv = 1e-5
                
                # Expand into successive fourier components until convergence or ic = nf
                
                for ic in range(nf+1):
                    ppln*=0
                    pmin*=0 
                    pplr*=0
                    pmir*=0

                    for j1 in range(ncont):                    
                        pfunc = phasarr[j1, widx, 0, :]
                        xmu   = phasarr[j1, widx, 1, :]
                        iscat = 4
                        ncons = 0
                        cons8 = pfunc
                        norm = 1
                        pplpl, pplmi, fc[j1] = calc_pmat6(ic, mu, wtmu, nmu, iscat, cons8, ncons, 
                                                  norm, j1, ncont, nphi, fc[j1], pfunc, xmu)
                        # Transfer matrices to those for each scattering particle
                        ppln[j1,:,:] = pplpl
                        pmin[j1,:,:] = pplmi

                    if iray > 0:
                        iscat = 0
                        ncons = 0
                        pplpl, pplmi, fc[ncont] = calc_pmat6(ic, mu, wtmu, nmu, iscat, cons8, ncons, 
                                                    1, ncont, ncont, nphi, fc[ncont], pfunc, xmu)

                        # Transfer matrices to storage
                        pplr[:,:] = pplpl
                        pmir[:,:] = pplmi

                    #Computing RTJ matrices for the surface
                    surface_defined = False
                    if( (lowbc == 1) & (lookdown==True) ):
                        
                        surface_defined = True
                        
                        jl[:,0] = (1-galb[widx])*radg[widx]
                        if ic == 0:
                            tl *= 0.0
                            for j in range(nmu):
                                rl[j,:] = 2*galb[widx]*mu[j]*wtmu[j] 
                                rl[:,:]*= xfac
                        else:
                            tl *= 0.0
                            rl *= 0.0
                        jbase = jl
                        rbase = rl
                        tbase = tl 

                    # Main loop: computing R,T,J for each layer
                    for l in range(0,nlay):
                        
                        if lookdown is True:
                            k = l
                        else:
                            k = nlay - 1 - l
                         
                        #Defining the properties of the layer
                        taut = tau[widx,k]
                        bc = bnu[widx,k]
                        omega = omegas[widx,k]
                        if omega < 0:
                            omega = 0.0
                        if omega > 1:
                            omega = 1.0

                        tauscat = taut*omega
                        taur = tauray[widx,k]
                        tauscat = tauscat-taur

                        if tauscat < 0:
                            tauscat = 0.0  
                            
                        omega = (tauscat+taur)/taut
                        frac = lfrac[widx,:,k]

                        #Calculating the matrices
                        rl, tl, jl, iscl = calc_rtj_matrix(ic,mu,wtmu,bc,taut,tauscat,taur,frac,ppln,pmin,pplr,pmir,cc=cc,mminv=mminv,e=e)
                        
                        #Combining layers along the path
                        if l == 0 and surface_defined == False:
                            jbase = jl
                            rbase = rl
                            tbase = tl
                        else:
                            rbase, tbase, jbase = addp(rl, tl, jl, iscl,
                                                       e, rbase, tbase, jbase, nmu)

                    if ic != 0:
                        jbase *= 0.0


                    #Calculating the spectrum
                    for j in range(nmu):
                        u0pl[j] = 0.0
                        if ic == 0:
                            utmi[j] = radg[widx,j]
                        else:
                            utmi[j] = 0.0

                    ico = 0

                    if lookdown is True:
                        for imu0 in range(isol, isol+2): 
                            u0pl[imu0] = solar1[widx] / (2.0 * np.pi * wtmu[imu0])
                            acom = np.ascontiguousarray(rbase.transpose()) @ u0pl
                            bcom = np.ascontiguousarray(tbase.transpose()) @ utmi
                            acom += bcom
                            umi = acom + jbase
                            for imu in range(iemm, iemm+2): 
                                yx[ico] = umi[imu,0]
                                ico += 1
                                u0pl[imu0] = 0.0 
                    else:
                        for imu0 in range(isol, isol+2): 
                            u0pl[imu0] = solar1[widx] / (2.0 * np.pi * wtmu[imu0])
                            acom = np.ascontiguousarray(tbase.transpose()) @ u0pl
                            bcom = np.ascontiguousarray(rbase.transpose()) @ utmi
                            acom += bcom
                            umi = acom + jbase
                            for imu in range(iemm, iemm+2): 
                                yx[ico] = umi[imu,0]
                                ico += 1
                                u0pl[imu0] = 0.0 

                    drad = ((1-t)*(1-u)*yx[0] + t*(1-u)*yx[1] + t*u*yx[3] + (1-t)*u*yx[2]) * np.cos(ic*aphi * np.pi / 180.0)

                    if ic > 0:
                        drad *= 2

                    rad[ipath,ig,widx] += drad
                    conv = np.abs(drad / rad[ipath,ig,widx])
                    
                    # If sure about convergence, exit loop over fourier components
                    if conv < defconv and conv1:
                        break

                    if conv < defconv:
                        conv1 = True
                    else:
                        conv1 = False
    
    return rad



