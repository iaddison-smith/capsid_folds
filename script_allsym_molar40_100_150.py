import numpy
from matplotlib import pyplot as plt
from matplotlib import rcParams
import os
import glob
import zika_postprocess as zp
plt.rcParams['xtick.labelsize']=8
plt.rcParams['ytick.labelsize']=8





# ### Read dphir.txt files from Pygbe

data_sim,data_sim1,data_sim2,data_sim3 = zp.get_data_sym(sym=0), zp.get_data_sym(sym=1), zp.get_data_sym(sym=2), zp.get_data_sym(sym=3)

for molar in ['40','100','150']:
    for sym_active in ['sym1']:

        print('Plots for')
        print('symmetry',sym_active)
        print('molar',molar)

        data_sim_selected = [data_sim1 if sym_active=='sym1' else data_sim2 if sym_active=='sym2' else data_sim3 if sym_active=='sym3' \
            else data_sim][0]
        pqr_file = ['zika_sym1(90_y-axis).pqr' if sym_active == 'sym1' else 'zika_sym2(146_y-axis).pqr' if sym_active == 'sym2'\
            else 'zika_sym3(128_y-axis).pqr'][0]

        dist,fqf,_ = zp.compute_force_qf_zika(data_sim_selected[molar], units='pN',fqf_calc=False)
        fdb, fib = zp.get_boundary_forces(data_sim_selected, molar)

        # Binding force by substracting solvation force for capsid far away from plane
        F_terms = fqf+fdb+fib - (fqf[-1]+fdb[-1]+fib[-1] )
        f_tot_mag_386 = numpy.sqrt(numpy.sum(F_terms[0]**2)) # Total force magnitude at 2 A distance
        f_tot_xaxis_386 = F_terms[0][0] # Total force magnitude at 2 A distance

        dphi_386c = numpy.loadtxt(glob.glob(data_sim_selected[molar][0] + '\*dphir.txt')[0])   # d=0.2nm
        #dphi_390c = numpy.loadtxt(folder+'390/dphir.txt')   # d=0.6nm
        #dphi_394c = numpy.loadtxt(folder+'394/dphir.txt')   # d=1.0nm
        #dphi_404c = numpy.loadtxt(folder+'404/dphir.txt')   # d=2.0nm
        #dphi_484c = numpy.loadtxt(folder+'484/dphir.txt')   # d=10.0nm
        dphi_1384c = numpy.loadtxt(glob.glob(data_sim_selected[molar][-1] + '\*dphir.txt')[0]) # d=100.0nm

        # Read pqr file and calculate forces
        N = len(dphi_386c)
        position, q, amino_acid_name, amino_acid_number, atom_name, chain = zp.readpqr(pqr_file,N)
        delta_dphi_386_local = dphi_386c - dphi_1384c #Calculate interaction potential
        fqf_386, fqf_mag_386, Fqf_386, Fqf_mag_386 = zp.compute_force_qf(delta_dphi_386_local, q, units='pN')

        # Identify aminoacids in .pqr file
        types = []
        for j in range(amino_acid_name.shape[0]):
            if amino_acid_name[j] not in types:
                types.append(amino_acid_name[j])




        plane = 0#-145.25
        near_plane = numpy.where(abs(position[:,2]-plane)<0.5)[0] # Atoms between +-0.5 A
        near_plane2 = numpy.where(abs(position[:,2]-plane)<2.5)[0] # Atoms between +-2.5 A

        # Plotting
        x = position[near_plane,0]
        y = position[near_plane,1]
        x2 = position[near_plane2,0]
        y2 = position[near_plane2,1]
        f_porcent = 100*(fqf_mag_386[near_plane])/(f_tot_mag_386) #Force percentage
        f_porcent2 = 100*(fqf_mag_386[near_plane2])/(f_tot_mag_386) #Force percentage
        f_greater  = numpy.where(f_porcent2 > 1)[0] #Detect forces greater than 1% of the total force
        fqf_plane = fqf_386[near_plane2]
        fqf_plane_greater = fqf_plane[f_greater]

        plt.clf()
        plt.figure(figsize=(4,3),dpi=150)
        plt.scatter(x2,y2,c=f_porcent2,s=0.5, cmap='cividis_r',vmin=-0.3)
        plt.scatter(x2[f_greater],y2[f_greater],c=f_porcent2[f_greater],cmap='cividis_r',s=0.5,vmin=0,vmax=1)
        plt.colorbar(label=r'$\bar{F}_{qf} /  \bar{F}_{bind} $ (%) ')
        plt.ylabel('$\AA$')
        plt.savefig('plots_capsid\\fqf_3a_z={0}_{1}_{2}mM.pdf'.format(plane,sym_active,molar), format='pdf',dpi=150,bbox_inches='tight')
        print('Plane z =', plane)
        print('Atoms with force greater than 1% of the total force: ', len(f_greater))
        print('Atoms displaced by more than +-0.5 Angstrom: ', len(near_plane))
        print('Atoms displaced by more than +-2.5 Angstrom: ', len(near_plane2))
        print('Total force for atoms greater than 1% of the total force: ', numpy.sum(fqf_plane_greater,axis=0))
        print('Total force in slice: ', numpy.sum(fqf_plane,axis=0))
        plt.close()



        plane = 210#-145.25
        near_plane = numpy.where(abs(position[:,0]-plane)<2.5)[0]

        # Plotting
        y = position[near_plane,1]
        z = position[near_plane,2]
        f_porcent = 100*(fqf_mag_386[near_plane])/(f_tot_mag_386)
        plt.clf()
        plt.figure(figsize=(4,3),dpi=150)
        plt.scatter(y,z,c=f_porcent,cmap='cividis_r',vmin=0,marker='.',s=0.5) # ver si quitar el marker
        plt.colorbar(label=r'$\bar{F}_{qf} /\bar{F}_{bind} $ (%) ')
        plt.xlim([-100,100])
        plt.ylabel('$\AA$')
        plt.savefig('plots_capsid\\fqf_3b_x={0}ang_{1}_{2}mM.pdf'.format(plane,sym_active,molar), format='pdf',dpi=150,bbox_inches='tight')
        print('Plane x =', plane)
        print('Atoms between +-2.5 Angstrom:', len(near_plane))
        plt.close()


        # Plotting for each chain
        plane = 210 #-145.25
        near_plane_abs = numpy.where(abs(position[:,0]-plane)<2.5)[0] #Plane between +-1 angstrom
        near_plane_GLU = numpy.where(amino_acid_name[near_plane_abs] == b'GLU')[0] #Selecting atoms of the chain GLU
        near_plane_LYS = numpy.where(amino_acid_name[near_plane_abs] == b'LYS')[0] #Selecting atoms of the chain LYS
        near_plane_ASP = numpy.where(amino_acid_name[near_plane_abs] == b'ASP')[0] #Selecting atoms of the chain ASP


        # Plotting
        y_GLU = position[near_plane_abs[near_plane_GLU],1]
        z_GLU = position[near_plane_abs[near_plane_GLU],2]
        f_porcent_GLU = 100*(fqf_mag_386[near_plane_abs[near_plane_GLU]])/(f_tot_mag_386)
        f_greater_GLU = numpy.where(f_porcent_GLU > 3.6)[0] #Detect forces greater than 3.6% of the total force
        fqf_plane_greater_GLU = fqf_386[near_plane_abs[near_plane_GLU[f_greater_GLU]]]

        y_LYS = position[near_plane_abs[near_plane_LYS],1]
        z_LYS = position[near_plane_abs[near_plane_LYS],2]
        f_porcent_LYS = 100*(fqf_mag_386[near_plane_abs[near_plane_LYS]])/(f_tot_mag_386)
        f_greater_LYS = numpy.where(f_porcent_LYS > 3.6)[0] #Detect forces greater than 3.6% of the total force
        fqf_plane_greater_LYS = fqf_386[near_plane_abs[near_plane_LYS[f_greater_LYS]]]

        y_ASP = position[near_plane_abs[near_plane_ASP],1]
        z_ASP = position[near_plane_abs[near_plane_ASP],2]
        f_porcent_ASP = 100*(fqf_mag_386[near_plane_abs[near_plane_ASP]])/(f_tot_mag_386)
        f_greater_ASP = numpy.where(f_porcent_ASP > 3.6)[0] #Detect forces greater than 3.6% of the total force
        fqf_plane_greater_ASP = fqf_386[near_plane_abs[near_plane_ASP[f_greater_ASP]]]

        plt.clf()
        plt.figure(figsize=(4,3),dpi=150)
        plt.scatter(y_GLU,z_GLU,c=f_porcent_GLU,s=25,cmap='cividis_r',marker='v',vmin=0,vmax=3.6,label='GLU')
        plt.scatter(y_ASP,z_ASP,c=f_porcent_ASP,s=25,cmap='cividis_r',marker='x',vmin=0,vmax=3.6,label='ASP')
        plt.scatter(y_LYS,z_LYS,c=f_porcent_LYS,s=25,cmap='cividis_r',marker='o',vmin=0,vmax=3.6,label='LYS')
        plt.xlim([-40,40]); plt.ylim([-40,40])
        plt.legend(loc='upper left',prop={'size': 6})
        plt.ylabel('$\AA$')
        plt.xlabel('$\AA$')
        plt.colorbar(label=r'$\bar{F}_{qf} / \bar{F}_{bind}  $ (%) ')
        plt.savefig('plots_capsid\\fqf_3c_x={0}ang_{1}_{2}_{3}mM.pdf'.format(plane,'GLULYSASP',sym_active,molar), format='pdf',dpi=150,bbox_inches='tight')
        plt.close()
        print('Plane x =', plane)
        print('Atoms displaced by more than +-2.5 Angstrom: ', len(near_plane_abs))
        print('Atoms of GLU greater than 3.6% in slice ', len(fqf_plane_greater_GLU))
        print('Total force for atoms of GLU greater than 3.6% of the total force: ', numpy.sum(fqf_plane_greater_GLU,axis=0))
        print('Atoms of LYS greater than 3.6% in slice ', len(fqf_plane_greater_LYS))
        print('Total force for atoms of LYS greater than 3.6% of the total force: ', numpy.sum(fqf_plane_greater_LYS,axis=0))
        print('Atoms of ASP greater than 3.6% in slice ', len(fqf_plane_greater_ASP))
        print('Total force for atoms of ASP greater than 3.6% of the total force: ', numpy.sum(fqf_plane_greater_ASP,axis=0))
        print('Total force in slice: ', numpy.sum(fqf_386[near_plane_abs] ,axis=0))


        h= 5
        planes, f_porcent_chain,f_porcent_chain_mag = zp.get_f_porcent_planes(0,235,h,fqf_386,types,position,amino_acid_name,f_tot_xaxis_386)

        # Verification of the most important amino acids by force magnitude
        f_porcent_chain_mag_sorted = sorted(f_porcent_chain_mag.items(), key=lambda x: max(abs(x[1])), reverse=True)   
        planes_fmax = []
        aminoacid_fmax = []
        for chain, forces in f_porcent_chain_mag_sorted:
            print('Max force in chain %s: %.2f, slice location %.1d' %(chain,max(abs(forces)),planes[numpy.argmax(abs(forces))]))
            planes_fmax.append(planes[numpy.argmax(abs(forces))])
            aminoacid_fmax.append(chain)

        plt.clf()
        plt.figure(figsize=(4,3),dpi=150)
        # Minus sign in forces to follow the convention of the paper
        plt.bar(planes, f_porcent_chain_mag[aminoacid_fmax[0]],width=h,label=aminoacid_fmax[0],color=zp.color_aminoacid(aminoacid_fmax[0])) 
        plt.bar(planes, f_porcent_chain_mag[aminoacid_fmax[1]],width=h,label=aminoacid_fmax[1],color=zp.color_aminoacid(aminoacid_fmax[1])) 
        plt.bar(planes, f_porcent_chain_mag[aminoacid_fmax[2]],width=h,label=aminoacid_fmax[2],color=zp.color_aminoacid(aminoacid_fmax[2])) 
        plt.bar(planes, f_porcent_chain_mag[aminoacid_fmax[3]],width=h,label=aminoacid_fmax[3],color=zp.color_aminoacid(aminoacid_fmax[3])) 
        plt.bar(planes, f_porcent_chain_mag[aminoacid_fmax[4]],width=h,label=aminoacid_fmax[4],color=zp.color_aminoacid(aminoacid_fmax[4]))
        plt.legend(prop={'size': 6})
        plt.xlabel('x $(\AA)$'); plt.ylabel(r'$F_{qf,x}/F_{bind,x}$ (%) ')
        plt.xlim([0,320]); plt.xticks(numpy.arange(0,350,50))
        plt.savefig('plots_capsid\\fqf_4a_forcesbar_h={0}_{1}_{2}mM.pdf'.format(str(h),sym_active,molar), format='pdf',dpi=150,bbox_inches='tight')
        plt.close()





        plt.clf()
        plt.figure(figsize=(4,3),dpi=150)
        plt.bar(planes-2*h/10, f_porcent_chain_mag[aminoacid_fmax[0]],width=h/10,label=aminoacid_fmax[0],color=zp.color_aminoacid(aminoacid_fmax[0])) 
        plt.bar(planes-h/10, f_porcent_chain_mag[aminoacid_fmax[1]],width=h/10,label=aminoacid_fmax[1],color=zp.color_aminoacid(aminoacid_fmax[1])) 
        plt.bar(planes, f_porcent_chain_mag[aminoacid_fmax[2]],width=h/10,label=aminoacid_fmax[2],color=zp.color_aminoacid(aminoacid_fmax[2]))
        plt.bar(planes+h/10, f_porcent_chain_mag[aminoacid_fmax[3]],width=h/10,label=aminoacid_fmax[3],color=zp.color_aminoacid(aminoacid_fmax[3])) 
        plt.bar(planes+2*h/10, f_porcent_chain_mag[aminoacid_fmax[4]],width=h/10,label=aminoacid_fmax[4],color=zp.color_aminoacid(aminoacid_fmax[4]))
        plt.legend(prop={'size': 6})
        plt.xlabel('x $(\AA)$')
        plt.ylabel(r'$F_{qf,x}/F_{bind,x}$ (%) ')
        plt.xticks(numpy.linspace(185,240,(240-185)//h +1))
        plt.xlim([182,232])
        plt.savefig('plots_capsid\\fqf_4b_forcesbar_h={0}_{1}_{2}mM.pdf'.format(h,sym_active,molar), format='pdf',dpi=150,bbox_inches='tight')
        plt.close()




        n_aminoacid = dict()
        for aminoacid in types:
            aminoacid_name = aminoacid.decode('utf-8')
            n_aminoacid[aminoacid_name] = len(numpy.where(aminoacid == amino_acid_name)[0])

        # sort by number of atoms in n_aminoacid
        n_aminoacid_sorted = sorted(n_aminoacid.items(), key=lambda x: x[1], reverse=True)   
        print('Total number of atoms:', len(atom_name))
        print('Total number of amino acids atoms:', sum(n_aminoacid.values()))
        n_aminoacid_sorted

        h= 5
        planes, n_atoms = zp.get_atoms_aminoacid_planes(0,235,h,types,position,amino_acid_name)

        plt.clf()
        plt.figure(figsize=(4,3),dpi=150)
        plt.bar(planes-2*h/10, n_atoms[aminoacid_fmax[0]],width=h/10,label=aminoacid_fmax[0],color=zp.color_aminoacid(aminoacid_fmax[0])) 
        plt.bar(planes-h/10, n_atoms[aminoacid_fmax[1]],width=h/10,label=aminoacid_fmax[1],color=zp.color_aminoacid(aminoacid_fmax[1])) 
        plt.bar(planes, n_atoms[aminoacid_fmax[2]],width=h/10,label=aminoacid_fmax[2],color=zp.color_aminoacid(aminoacid_fmax[2])) 
        plt.bar(planes+h/10, n_atoms[aminoacid_fmax[3]],width=h/10,label=aminoacid_fmax[3],color=zp.color_aminoacid(aminoacid_fmax[3])) 
        plt.bar(planes+2*h/10, n_atoms[aminoacid_fmax[4]],width=h/10,label=aminoacid_fmax[4],color=zp.color_aminoacid(aminoacid_fmax[4]))
        plt.legend(prop={'size': 6})
        plt.xlabel('x $(\AA)$')
        plt.ylabel(r'N° of atoms')
        plt.xticks(numpy.linspace(185,240,(240-185)//h +1))
        plt.xlim([182,232])
        plt.ylim([0,1800])
        plt.savefig('plots_capsid\\fqf_4c_atomsbar_h={0}_{1}_{2}mM.pdf'.format(h,sym_active,molar), format='pdf',dpi=150,bbox_inches='tight')
        plt.close()


        plt.clf()
        plt.figure(figsize=(4,3),dpi=150)
        plt.bar(planes-2*h/10, (f_porcent_chain_mag[aminoacid_fmax[0]] / n_atoms[aminoacid_fmax[0]]) ,width=h/10,label=aminoacid_fmax[0],color=zp.color_aminoacid(aminoacid_fmax[0])) 
        plt.bar(planes-h/10, f_porcent_chain_mag[aminoacid_fmax[1]]/n_atoms[aminoacid_fmax[1]],width=h/10,label=aminoacid_fmax[1],color=zp.color_aminoacid(aminoacid_fmax[1])) 
        plt.bar(planes, f_porcent_chain_mag[aminoacid_fmax[2]]/n_atoms[aminoacid_fmax[2]],width=h/10,label=aminoacid_fmax[2],color=zp.color_aminoacid(aminoacid_fmax[2]))
        plt.bar(planes+h/10, f_porcent_chain_mag[aminoacid_fmax[3]]/n_atoms[aminoacid_fmax[3]],width=h/10,label=aminoacid_fmax[3],color=zp.color_aminoacid(aminoacid_fmax[3])) 
        plt.bar(planes+2*h/10, f_porcent_chain_mag[aminoacid_fmax[4]]/n_atoms[aminoacid_fmax[4]],width=h/10,label=aminoacid_fmax[4],color=zp.color_aminoacid(aminoacid_fmax[4]))
        plt.legend(prop={'size': 6})
        plt.xlabel('x $(\AA)$')
        plt.ylabel(r'$F_{qf,x}/F_{bind,x}$ (%) ')
        plt.xticks(numpy.linspace(185,240,(240-185)//h +1))
        plt.xlim([182,232])
        plt.savefig('plots_capsid\\fqf_4c_forceatomsbar_h={0}_{1}_{2}mM.pdf'.format(h,sym_active,molar), format='pdf',dpi=150,bbox_inches='tight')
        plt.close()



        planes, qnet_chain = zp.get_qnet_planes(-285,285,h,q,types,position,amino_acid_name)
        qnet_chain_sorted = sorted(qnet_chain.items(), key=lambda x: max(x[1]), reverse=True)   
        for chain, charge in qnet_chain_sorted:
            print('Max charge in chain %s: %.2f, slice location %.1d' %(chain,max(charge),planes[numpy.argmax(charge)]))

        for j in range(len(planes_fmax)):
            index = numpy.where(planes == planes_fmax[j])[0][0]
            print('Max charge in chain %s: %.2f, slice location %.1d' %(aminoacid_fmax[j],qnet_chain[aminoacid_fmax[j]][index],planes[index]))