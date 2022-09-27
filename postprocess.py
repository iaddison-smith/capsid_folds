#Importing modules and font style
import numpy as np
import os
import matplotlib.pyplot as plt
import glob
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 16
plt.rcParams['xtick.labelsize']=16
plt.rcParams['ytick.labelsize']=16
plt.rcParams['lines.linewidth']=3
plt.rcParams['lines.markersize']=10

molar_data = [2,5,10,  20,  30,  40,  50,  60,  70,  80,  90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200] 

kappa = np.array([0.01451757, 0.0229543 , 0.03246228, 0.04590859, 
        0.05622631, 0.06492456, 0.07258786, 0.07951602, 0.08588711, 
        0.09181719, 0.09738683, 0.10265474, 0.1076652 , 0.11245263, 
        0.11704441, 0.12146272, 0.12572586, 0.12984911, 0.1338454 , 
        0.13772578, 0.14149979, 0.14517572])

molar_to_kappa = dict(zip(molar_data, kappa))

def get_energy_data(results_file_path):
    """
    Read energy and distances from Pygbe results file, for 
    a afm-tip example
    -------
    Inputs:
        results_file_path: Pygbe results file
    Return:
        None
    """
    results_file = open(results_file_path,'r')
    results_data = results_file.read().split('\n')
    for line in results_data:
        if 'Surface 0' in line:
            surf = 0
        elif 'Surface 1' in line:
            surf = 1
        elif 'box center' in line:
            aux = line.split(': ')
            if surf == 0:
                line = aux[1].split(',')
                r_surf1 = np.array([float(line[0]),float(line[1]),float(line[2])])
            elif surf == 1:
                line = aux[1].split(',')
                r_surf2 = np.array([float(line[0]),float(line[1]),float(line[2])])
    aux = results_data[-6].split('= ')
    E_solv = float(aux[1].split(' ')[0])
    aux = results_data[-5].split('= ')
    E_surf = float(aux[1].split(' ')[0])
    aux = results_data[-4].split('= ')
    E_coul = float(aux[1].split(' ')[0])
    results_file.close()
    
    return r_surf1, r_surf2, E_solv, E_surf, E_coul

def readpqr(filename, N):
    """
    Read pqr-format file
    -------
    Inputs:
        filename: file .pqr with point-charge-radius format
        N: number of solute charges present in filename
    Return:
        position: Array size (Nx3) with charges positions
        q: Array size (Nx1) with charges values
        amino_acid_name: Array (Nx1) with charges amino acid name
        amino_acid_number: Array (Nx1) with charges amino acid name 
        atom_name: Array (Nx1) with atom_names
        atom_number: Array (Nx1) with atom_numbers  
    """
    pqr_file = open(filename,'r')
    position = np.zeros((N,3))
    q = np.zeros(N)
    amino_acid_name = np.array(q, dtype=np.string_)
    atom_name = np.array(q, dtype=np.string_)
    chain = np.array(q, dtype=np.string_)
    amino_acid_number = np.zeros(N, dtype=int)
    counter = 0
    for i, line in enumerate(pqr_file):
        line_split = line.split()
        if line_split[0] == 'ATOM':
            position[counter,0] = float(line_split[6])
            position[counter,1] = float(line_split[7])
            position[counter,2] = float(line_split[8])
            q[counter] = float(line_split[9])
            amino_acid_name[counter] = line_split[3]
            amino_acid_number[counter] = int(line_split[5])
            atom_name[counter] = line_split[2]
            chain[counter] = line_split[4]
            counter += 1
        
    return position, q, amino_acid_name, amino_acid_number, atom_name,chain

def compute_force_qf(dphi, q, units='kcal'):
    """
    Compute forces due solute charges in the solute
    from dphi.txt file and q charges array
    -------
    Inputs:
        dphi: file .pqr with point-charge-radius format
        q: number of solute charges present in filename
    Return:
        force: Array size (Nx3) with forces for solute charges
        force_magnitude: Array size (Nx1) with forces magnitude for solute charges 
        total_force: Array size (1x3) with total force due solute charges
        total_force_magnitude: Array (1x1) with total force magnitude
    """
    if units == 'kcal':
        factor = 4*np.pi*332.0636817823836 #1e-3*Na*1e10*(qe**2/(ep_vacc*4*numpy.pi*cal2J))
    elif units == 'kJ':
        factor = 4.184*4*np.pi*332.0636817823836
    elif units == 'pN':
        factor = 69.467*4*np.pi*332.0636817823836
        
    force = -np.transpose(np.transpose(dphi)*q)
    force_magnitude = np.sqrt(np.sum(force**2, axis=1))
    total_force = np.sum(force, axis=0)
    total_force_magnitude = np.sqrt(np.sum(total_force**2))

    return factor*force, factor*force_magnitude, factor*total_force, factor*total_force_magnitude

def compute_force_qf_zika(dir, units='kcal',fqf_calc=True):
    """
    Read from directory with results file to get 
    fixed charge forces for afm-zika case
    -------
    Inputs:
        dir: directory with results file
        units: units for output forces (kcal/molA, kJ/molA, pN)
        fqf_calc: if True, compute forces from dphi.txt file (Slow)
    Return:
        fqf: Array size (1x3) with total force due solute charges
        fqf_mag: Array (1x1) with total force magnitude
    """    
    fqf = np.zeros((len(dir),3))
    fqf_mag = np.zeros(len(dir))
    dist = np.zeros((len(dir)))
    for j in range(len(dir)):
        dist[j] = (dir[j].split('dist')[-1])
        if fqf_calc:
            dphir_file = glob.glob(dir[j] + '\*dphir.txt')
            dphir = np.loadtxt(dphir_file[0])
            _, q, _, _, _,_ = readpqr('pqr\\ZIKV_6CO8_aa_charge_vdw_addspace.pqr',len(dphir))
            _, _, fqf[j,:], fqf_mag[j] = compute_force_qf(dphir,q, units)
            np.savetxt(dir[j] + '\\fqf.txt', fqf[j,:])
        else:
            fqf[j,:] = np.loadtxt(dir[j] + '\\fqf.txt')
            fqf_mag[j] = np.sqrt(np.sum(fqf[j,:]**2))

    return dist, fqf, fqf_mag

def get_boundary_forces(data_sim, molarity):
    """
    Extract boundary forces from pygbe results file
    -------
    Inputs:
        data_sim: dictionary with simulation data directory
        molarity: int value of molarity
    Return:
        fdb: Array size (3x1) with dielectric boundary force
        fib: Array size (3x1) with ionic boundary force
    """
    # Use the first distance (2 Ang) to search boundary forces
    files = glob.glob(data_sim[str(molarity)][0]+'\*')

    #Get only the bound files from files list
    files_bound = [file for file in files if 'Bound' in file.split('\\')[-1].split('_')[0]]
    files_bound = sorted(files_bound, key=lambda x: int(x.split('_')[-1].split('Plane')[0]))

    fdb, fib = np.zeros((len(files_bound),3)), np.zeros((len(files_bound),3))
    j = 0
    factor = 69.467*4*np.pi*332.0636817823836 #factor to convert to pN
    for file in files_bound:
        file_data = open(file,'r')
        file_data = file_data.read().split('\n')
        #Dielectric boundary force
        fdb[j,:] = factor*np.array(file_data[-4].split(' '),dtype=np.float64)
        #Ionic boundary force
        fib[j,:] = factor*np.array(file_data[-2].split(' '),dtype=np.float64)
        j += 1

    return fdb, fib

# MVM fitting function (Podgornik et al. Nanoscale, 2015)
def forces_MVM(d,B,A=0,k=1/6.8):
    return (A*np.exp(-2*k*d)+B*np.exp(-k*d))/(1-np.exp(-2*k*d))

def get_data_sym(sym=None):
    """
    Get directory who has simulation data
    -------
    Inputs:
        sym: string with symbol to search in file
    Return:
        data_sim: dictionary with keys as molarity and values as list of directories
    """
    
    data_sim = dict()
    if sym == 0:
        sim_data = glob.glob('allSimsZikaAsItIs/*')
    elif sym == 1:
        sim_data = glob.glob('ZikaSym1/*')
    elif sym == 2:
        sim_data = glob.glob('ZikaSym2/*')
    elif sym == 3:
        sim_data = glob.glob('ZikaSym3/*')
    else:
        print('Error: symmetry not valid')
        return None
    for dir in sim_data:
        molarity = dir.split('sysBunch')[1]
        data_sim[molarity] = glob.glob(dir+'/*')

    #sort a data_sim dictionary by key value
    data_sim = dict(sorted(data_sim.items(), key=lambda x: int(x[0])))
    #sort data_sim value array by dist value
    for key, value in data_sim.items():
        data_sim[key] = sorted(value, key=lambda x: int(x.split('dist')[-1]))
    return data_sim

def plot_force_components(data_sim,molar,sym=0,fqf_calc=True):

    fdb, fib = get_boundary_forces(data_sim, molar)
    dist,fqf,_ = compute_force_qf_zika(data_sim[molar], units='pN', fqf_calc=fqf_calc)

    # Binding force by substracting solvation force for capsid far away from tip
    F_terms = fqf+fdb+fib - (fqf[-1]+fdb[-1]+fib[-1] )

    delta_fqf = (fqf.swapaxes(0,1)[0]-fqf.swapaxes(0,1)[0][-1])
    delta_fdb = (fdb.swapaxes(0,1)[0]-fdb.swapaxes(0,1)[0][-1])
    delta_fib = (fib.swapaxes(0,1)[0]-fib.swapaxes(0,1)[0][-1])
    dd = np.arange(394,434,0.1)

    plt.figure(figsize=(10,6),dpi=75)
    plt.grid()
    plt.plot(dist,-delta_fqf, marker='D',label='$F_{qf}$')
    plt.plot(dist,-delta_fdb, marker='s',label='$F_{db}$')
    plt.plot(dist,-delta_fib, marker='o',label='$F_{ib}$')
    plt.plot(dist,-F_terms.swapaxes(0,1)[0], marker='x',label='$F_{total}$')
    plt.plot(dd[1:]-384,forces_MVM(dd[1:]-384,2,k=molar_to_kappa[int(molar)]),label='MVM B=2',linestyle='--')
    plt.grid()
    plt.xlim([0,20])
    plt.xlabel('d ($\AA$)');
    plt.ylabel('$F_{bind}$ (pN)')
    plt.legend()
    plt.title('Interaction force between capsid and plane for %d mM' % int(molar))
    if sym == 0:
        plt.savefig('plots\\interaction_force_afmtip_%d(mM).png' % int(molar),dpi=150, bbox_inches='tight')
    else:
       plt.savefig('plots\\interaction_force_afmtip_%d(mM)_sym%d.png' % (int(molar),sym),dpi=150, bbox_inches='tight') 
    plt.close()

    return None