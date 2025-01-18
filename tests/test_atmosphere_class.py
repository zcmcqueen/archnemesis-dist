import pytest  
import archnemesis as ans
import numpy as np
  
def test_gravity():  
    '''
    Test calculation of gravity
    '''
    
    #Declaring class
    Atmosphere = ans.Atmosphere_0()
    Atmosphere.IPLANET=3   #Earth - The test can fail if the parameters for Earth in the dictionary are changed
    Atmosphere.NP = 3
    Atmosphere.H = np.linspace(0.,30000.,Atmosphere.NP)
    Atmosphere.LATITUDE=0.
    Atmosphere.calc_grav()
    
    expected = np.array([9.78027361, 9.7341945, 9.68843803])
    # Use a NumPy comparison for arrays
    assert np.allclose(Atmosphere.GRAV, expected, atol=1e-6)
 
  
def test_molwt():
    '''
    Test calculation of mean molecular weight
    '''
    
    #Declaring class
    Atmosphere = ans.Atmosphere_0()

    Atmosphere.NP = 2
    Atmosphere.NVMR = 5
    Atmosphere.ID = np.array([1,2,3,4,5])
    Atmosphere.ISO = np.array([0,0,0,0,0])
    Atmosphere.AMFORM = 2   #Calculate molecular weight automatically
    
    vmr = np.zeros((Atmosphere.NP,Atmosphere.NVMR))
    vmr[0,:] = np.array([0.1,0.5,0.1,0.1,0.2])
    vmr[1,:] = np.array([0.5,0.1,0.1,0.2,0.1])
    Atmosphere.edit_VMR(vmr)
    
    Atmosphere.calc_molwt()
    
    expected = np.array([0.0386093, 0.02981164])
    # Use a NumPy comparison for arrays
    assert np.allclose(Atmosphere.MOLWT, expected, atol=1e-6)
    
    
