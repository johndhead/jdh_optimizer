#####################################################################################
#   Start of jdh_optimizer  24-dec-2019 version
#   version == "masswt" has  mass wt init_invhess
######################################################################################
# # Setup bfgs routine - from the Numerical Recipes book  6-dec-2019
# # clean_v2 - trying to improve optimization summary  12-dec-2019
# # routines rearranged to a python form 
# # that can be imported into psi4.driver.py routine 21-dec-2019
# # changing args in en_gr_fun to include energy/grad calculation options 21-dec-2019
# # Perform BFGS minimization on CO calc with HF/6-31g energy

# In[1]:


import psi4
import numpy as np
import sys
import os

# In[2]:


def print_mol_coord_sum(mol,opt_stage="optimized"):
    """ function to print summary of "opt_stage" mol coordinates """
    #print("\n================= Summary of %s %s coordinates in %s =========================\n" 
    #      % (opt_stage,mol.name(),mol.units()))
    psi4.core.print_out("\n================= Summary of %s %s coordinates in %s =========================\n" 
          % (opt_stage,mol.name(),mol.units()))

    for iat in range(mol.natom()):
        #print("    atom %d %3s %9.5f xyz coord = %15.9f %15.9f %15.9f" % (iat,mol.symbol(iat),mol.mass(iat),
        #      mol.xyz(iat)[0],mol.xyz(iat)[1],mol.xyz(iat)[2]))        
        psi4.core.print_out("\n    atom %d %3s %9.5f xyz coord = %15.9f %15.9f %15.9f" % (iat,mol.symbol(iat),mol.mass(iat),
              mol.xyz(iat)[0],mol.xyz(iat)[1],mol.xyz(iat)[2]))        
    #print("\n=======================================================================================")
    psi4.core.print_out("\n=======================================================================================\n")
    return

# In[3]:
#  set up functions to be used in the jdh_optimization routine
#  functions are (1) en_gr_fun - calcs energy and gradient at coords - where actual mol coords = init_coords + coords
#         (2) linsrch   and (3) dfpmin

################################################################
# set up energy_grad function
def en_gr_fun(coords,*args):
    """ en_gr_fun - calcs energy and gradient for mol
        coords = coords to calculate the energy - initially set to zero
        mol_coords = init_coords + coords
    
    args = (mol, eg_opts, init_coords, init_com, inv_sqrt_mass, coord_type)
    where:
        mol = molecule class name
        eg_opts = options for energy and grad calcs
        init_coords such that coords are zero initially
        init_com = initial center of mass when coords = zero
        inv_sqrt_mass = 1/sqrt(atmas[iat]) - used when coord_type = 'masswt'
        coord_type posibilities so far: 'cart'
    
    function returns scf_e and grad
        """
    
    #print("<<<<<<<< en_gr_fun coords: ",coords)
    #print("no of args in *args = %d" % len(args))
    if len(args) == 6:
        (mol,eg_opts,init_coords,init_com,inv_sqrt_mass,coord_type) = args
        #print("mol =",mol)
        print("in en_gr_fun: mol =",mol)
        #print("dir(mol): ->\n",dir(mol))
        #print("init_coords =",init_coords)
        #print("inv_sqrt_mass =",inv_sqrt_mass)
        #print("coord_type = %s"% coord_type)
        
    nat = mol.natom()
    if coord_type == 'cart':
        # coords equal linear array len 3*mol.natom()
        #debug -- print("cart disp coords: ",coords)
        pass
 
    elif coord_type == 'masswt':
        # coords are mass weighted - convert to cartessian
        coords = coords * inv_sqrt_mass # cartesian displacment
        
        #debug -- print("masswt disp coords as cart: ",coords)
    
    else:
        print("*** Error not set up for coord_type = %s" % coord_type)
        sys.exit()
        
        
    geom = np.reshape(coords,(nat,3)) + init_coords
        
    mol.set_full_geometry(psi4.core.Matrix.from_array(geom))
    
    #mol.update_geometry
    mol.update_geometry()

    #psi4.core.print_out("new mol geom: \n")
    print_mol_coord_sum(mol,opt_stage="New CART")
    
    #for iat in range(mol.natom()):
        #print("atom %d %3s %9.5f xyz coord = " % (iat,mol.symbol(iat),mol.mass(iat)),mol.xyz(iat))

    #print("\n===== co bond distance = %10.5f a.u." % (mol.z(1)-mol.z(0)))
    
    cxcom = mol.center_of_mass()[0]
    cycom = mol.center_of_mass()[1]
    czcom = mol.center_of_mass()[2]
    #print("cxcom,cycom,czcom: ",cxcom,cycom,czcom)
    current_com = np.array([cxcom,cycom,czcom],dtype=float)
    com_dif = current_com - init_com
    psi4.core.print_out("\n         ++++ current com = %18.10f  %18.10f  %18.10f" 
             % (current_com[0],current_com[1],current_com[2]))
    psi4.core.print_out("\n ++++  diff = curr - init = %18.10f  %18.10f  %18.10f a.u.\n"
          % (com_dif[0],com_dif[1],com_dif[2]))

    # get inertia tensor and rotational consts
    inert_ten = np.array(mol.inertia_tensor())
    cur_rotc = np.array(mol.rotational_constants())
    psi4.core.print_out("\ncurrent rot consts:  %15.9f  %15.9f  %15.9f" % (cur_rotc[0],cur_rotc[1],cur_rotc[2]))
    psi4.core.print_out("\ninert_ten -->\n")
    psi4.core.print_out(str(inert_ten))
    # calc evals and evecs for inertia_tensor
    teval,tevec = np.linalg.eigh(inert_ten)
    psi4.core.print_out("\n  Eigen vals and vecs from inertia tensor")
    for ivec in range(3):
        psi4.core.print_out("\neval[%d] = %12.8f  vec = (%11.8f, %11.8f, %11.8f)"
             % (ivec,teval[ivec],tevec[ivec,0],tevec[ivec,1],tevec[ivec,2]))

    scf_e,wavefn = psi4.energy(eg_opts,return_wfn=True)
    psi4.core.print_out("++++++++ scf_e in en_fun = %18.9f" % scf_e)
    #print("++++++++ scf_e in en_fun = %18.9f" % scf_e)
    
    G0 = psi4.gradient(eg_opts,ref_wfn=wavefn)
    gvec = np.array(G0)
    #jdhd - usually comment out this line 21-dec-2019
    #print("+=+=+=+=+ Cart gradient vector: \n", gvec)
    
    grad = np.reshape(gvec,(len(coords),))
    
    if coord_type == "masswt":
        grad *= inv_sqrt_mass
        #print("=+=+=+=+ Mass wt grad vector: \n",grad) 
    
    
    
    #print("+=+=+=+ grad as a linear array -->",grad)
    #print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    return scf_e, grad

####################### end en_gr_fun #################################
    


# In[4]:


##########################################################################################
# set up line search routine from section 9.7 from the numerical reciples book 6-dec-2019
def linsrch(func, xold, fold, g, p, dfp_iter, stpmax = 0.4, args=None):
    """ xold = init coords,
        fold and g = energy, gradient at xold
        p = search direction = xi in calling function dfpmin
        x = new coords along p = xold + alam * xi
        f = new energy
        stpmax = parameter for max step along p
        dft_iter = iter number in dftmin function
        
        On exit: check = False on normal exit
                       = True when x too close to xold - usually suggest optimization converged
        """
    check = False
    step_rec = []
    # constants
    ALF = 1.0e-4 
    TOLX = 1.0e-13  # this value is much greater than np.finfo()
    NPTS = 10  # max number of line searches
    
    # get norm of search direction
    sum  = np.sqrt(np.dot(p,p))
    
    psi4.core.print_out("\nLSEARCH: initial step size =%12.6f  stpmax = %f9.4 -- dfp_iter = %d\n" 
          % (sum,stpmax,dfp_iter))

    # scale p if step too big
    if sum > stpmax:
        p *= stpmax/sum
        
    # slope
    slope = np.dot(g,p)
    
    if slope >= 0.:
        psi4.core.print_out("\nLSEARCH: slope = %f which gt zero - Roundoff problem in lnsrch - dfp_iter = %d"
              % (slope,dfp_iter))
        print("LSEARCH: summary of linsrch - step_rec:\n",step_rec)
        return "No good"
    
    test = 0.0
    for i in range(len(xold)):
        #den = np.abs(xold[i])
        #if den > 1.:
        #    den = 1.
        #elif den < 0.01:
        #    den = 0.01
        #    if np.abs(p[i]) > 0.1:
        #        print("jdh warning in linsrch: i, xold[i], p[i], den =",i,xold[i],p[i],den)
        #        print("jdh: xold = ",xold)
        #        print(" going to give unreasonable temp = abs(p[i])/den = ",np.abs(p[i])/den,"  - CHECK")

        #temp = np.abs(p[i])/den
        abs_xold = np.abs(xold[i])
        if abs_xold < 1.0:
            abs_xold = 1.0
        temp = np.abs(p[i])/abs_xold
        if temp > test:
            test = temp
    
    alamin = TOLX/test
    alam = 1.0   # Always try full newton step first
    for istep in range(NPTS):
        #print("istep in linsrch = ",istep)
        x = xold + alam * p
        f,newgrad = func(x,*args)
        step_rec.append([dfp_iter,istep,alam,f])
        psi4.core.print_out("\nLSEARH: dfp_iter = %d  istep = %d  alam = %f" % (dfp_iter,istep,alam))
        psi4.core.print_out("\nLSEARCH === energy value at step = %13.7f" % f)
        psi4.core.print_out("\nLSEARCH newgrad = \n")
        psi4.core.print_out(str(newgrad))
        if alam < alamin:
            x = xold
            check = True
            psi4.core.print_out("\nLSEARCH: alam < alamin = %f - quit line search with check = True" % alamin)
            return check,x,f,newgrad,step_rec
        elif f <= fold + ALF*alam*slope:
            psi4.core.print_out("\nLSEARCH: worked good - check set = False  dfp_iter = %d linsrch step = %d"
                 % (dfp_iter,istep))
            return check,x,f,newgrad,step_rec   # line search worked OK
        else:  # step too big and needs to backtrack
            if alam == 1.0:  # first step too big
                tmplam = -slope/(2.0*(f-fold-slope))
            else:  # step obtained by fitting to cubic
                rhs1 = f - fold - alam*slope
                rhs2 = f2 - fold - alam2*slope
                a = (rhs1/(alam*alam) - rhs2/(alam2*alam2) )/(alam-alam2)
                b = (-alam2*rhs1/(alam*alam) + alam*rhs2/(alam2*alam2))/(alam-alam2)
                if a == 0.:
                    tmplam = -slope/(2.0*b)
                else:
                    disc = b*b - 3.0 *a*slope
                    if disc < 0.0:
                        tmplam = 0.5 * alam
                    elif b <= 0.0:
                        tmplam = (-b + np.sqrt(disc))/(3.0*a)
                    else:
                        tmplam = -slope/(b+np.sqrt(disc))
                    
                    if tmplam  > 0.5 * alam:
                        tmplam = 0.5 * alam
        
        alam2 = alam
        f2 = f
        psi4.core.print_out("\nLSEARCH: Find max of tmplam %f and 0.1*alam %f" % (tmplam,0.1*alam))
        if tmplam > 0.1*alam:
            alam = tmplam
        else:
            alam = 0.1 * alam
        #alam = np.max(tmplam,0.1*alam)
        psi4.core.print_out("\nLSEARCH === shorten step size: istep = %d  alam reset to %f -- dfp_iter = %d "
              % (istep,alam,dfp_iter))
        

    psi4.core.print_out("\nLSEARCH +=+=+ end of linsrch routine +=+=+ SHOULD NOT REACH HERE -- dfp_iter = %d" % dfp_iter)
                    
######################### end linsrch ####################################
#


# In[5]:


#
################################ start of dfpmin ######################################
# numerical recipes quasinewton program from section 10.9 of the box
# quasinewton uses the linsrch function above and the bfgs update for an inverse hessian
# 7-dec-2019

def dfpmin(func, p, args=None, itmax=100, stpmx=0.4, etol= 1.e-6, gtol=1.e-4, xtol=1.e-3,init_invhess = None):
    """  function dfpmin finds the local minimum for the function funcd 
         which computes an energy and energy derivative for some molecule
         
         p = initial molecules coordinate
         
         args = (mol, eg_opts, init_coords, init_com, inv_sqrt_mass, coord_type)
    where:
        mol = molecule class name
        eg_opts = energy/gradient options
        init_coords such that coords are zero 
        init_com = mol center of mass at init_coords
        inv_sqrt_mass = 1/sqrt(atmas[iat]) - used when coord_type = 'masswt'
        coord_type posibilities so far: 'cart'
        
        maxiter = no of geometry searches performed
        stpmx = max step size in individual geometry change
        etol = tolerance on desired energy change at min = fp(iter) - f(iter-1)
        gtol = tolerance on max gradient at converged minimum
        xtol = tolerance on max step size at converged minimum
        init_invhess = enables a guess at the starting invhess used in the optimization
        
     dfmin returns opt_code, fret, p_opt, gr_opt, opt_tol, invhess
     where: 
         opt_code = ?
         fret = energy value of optimized geom
         p_opt = optimized coordinates
         gr_opt = energy gradients at optimized geom
         opt_tol = [???] summarizing final opt tolerances
         invhess = final approximate invhess
         
    """
    if args is None:
        coord_type = None
    else:
        mol = args[0]
        coord_type = args[-1]
        if coord_type == "masswt":
            inv_sqrt_mass = args[-2]
            car_xtest = 0.
            car_gtest = 0.
    
    EPS = 1.e-14
    # get number of parameters to be optimized - typical nvar = 3*(no atoms)
    nvar = len(p)
    # get initial energy and gradient
    # set up a list with summary of optimization
    dfpmin_iter_sum = []
    # contents of iter_record
    # iter_record = [iter,energy,coords,grad,step,del_en,xtest,gtest,e_conv,x_conv,g_conv,hessin]
    iter = 0
    fp,g = func(p,*args)
    psi4.core.print_out("\n@Iter %3d: Energy %15.9f and grad -->\n" % (iter,fp))
    psi4.core.print_out(str(g))
    
    # setup initial hessian - eventually should have various options for this
    if init_invhess is None:
        hessin = np.identity(nvar,dtype=float)
    elif coord_type == "masswt":
        hessin = init_invhess
        psi4.core.print_out("\n coord_type = %s and init_invhess is mass weighted unit matrix\n" % coord_type)
    else:
        print("init_invhess = ",init_invhess)
        print("function dfpmin is not currently setup for this option - pgm quits")
        sys.exit()
        
    # check initial point is not a stationary point

    g_norm = np.sqrt(np.dot(g,g))

    # get max gnew component
    gtest = np.max(np.abs(g))
    psi4.core.print_out("\n@@@ At initial point max g component = %13.6e and norm = %13.6e" % (gtest,g_norm))
    
    iter_record = [iter,fp,p,g,0.,fp,0.,gtest,False,False,gtest < gtol] # could add hessin
    totno_en_gr_evals = 1
    if coord_type == "masswt":
        iter_record.append(0.)
        car_gtest = np.max(np.abs(g/inv_sqrt_mass))
        iter_record.append(car_gtest)
    
    dfpmin_iter_sum.append(iter_record)

    if gtest < gtol:
        psi4.core.print_out("\n@@@ Initial geometry is a stationary point: max g component < gtol = %13.6e" % gtol)
        psi4.core.print_out("\n@@@ Exit dfpmin routine")
        return fp,p,g,hessin
    
    # Now start iterating over steps
    for iter in range(1,itmax):
        
        # get step direction
        xi = - np.dot(hessin,g)
        xi_norm = np.sqrt(np.dot(xi,xi))
        psi4.core.print_out("\n@Iter %3d: norm = %9.5f Step:" % (iter,xi_norm))
        psi4.core.print_out(str(xi))
        
        check,pnew,fret,gnew,linsrch_sum = linsrch(en_gr_fun,p,fp,g,xi,iter,stpmx,args=args)
        if check:
            print("**WARNING** linsrch check = True - there is a problem with the linsrch")
        if len(linsrch_sum) > 1:
            # linsrch_sum formed by appending [istep,alam,f] into step_rec.append([dfp_iter,istep,alam,f])
            
            for linsrch_rec in range(len(linsrch_sum)):
                dfpmin_iter_sum.append(linsrch_sum[linsrch_rec])
                
        # keep track of number of energy evals - including those in linsrch routine
        totno_en_gr_evals += len(linsrch_sum)
                
            
        #linsrch(en_gr_fun,coords,e,grad,step,stpmax,args=(co, init_coords,inv_sqrt_mass,"cart"))
        
        del_en = fp - fret
        psi4.core.print_out("\n@Iter %3d after linsrch New E = %15.9f  Delta E = %15.9f  linsrch check = %s"
               % (iter,fret,del_en,check))
        psi4.core.print_out("\n@@ pnew = \n")
        psi4.core.print_out(str(pnew))
        psi4.core.print_out("\n@@ gnew = \n")
        psi4.core.print_out(str(gnew))
        
        fp = fret
        # save initial xi as old_xi and new step from linsrch
        old_xi = xi.copy()
        xi = pnew - p
        #print("step-xi = ",xi)

        # note xi = geometry step = (x_(i+1) - x_i) - following print checks xi correct
        #print("+++ if alam = 1.0 in LINSRCH then xi - oldxi should == zero -->\n",(xi - old_xi))

        # update the current point p
        p = pnew.copy()  # need to check this bit carefully
        
        # check size of next step xi
        xi_norm = np.sqrt(np.dot(xi,xi))
        
        # get max xi componets
        xtest = np.max(np.abs(xi))
        if coord_type == "masswt":
            car_xtest = np.max(np.abs(xi*inv_sqrt_mass))
            psi4.core.print_out("\n@@@ max xi step component = %10.6f and norm = %10.6f  car_xtest = %10.6f" % (xtest,xi_norm,car_xtest))
        else:
            psi4.core.print_out("\n@@@ max xi step component = %10.6f and norm = %10.6f" % (xtest,xi_norm))
        
        e_conv = False
        x_conv = False
        g_conv = False
        
        # Test to see local min energy converged within etol threshold
        if del_en < 0.:
            print("\n@Iter %3d new energy higher than old energy - linsrch not working correctly ERROR??" % iter)
            print("\n@Iter %3d old energy = %15.9f  new energy = %15.9f" % (iter,fp,fret))
        elif del_en < etol:
            psi4.core.print_out("\n@Iter %d - optimization converged on energy" % iter)
            e_conv = True
        
        # Test to see if local min geometry found within xtol threshold
        if xtest < xtol:
            psi4.core.print_out("\n@Iter %3d - optimization converged on xtol " % iter)
            x_conv = True
            #return fp,p,gnew
        
        # Opt not converged on xtol
        # Test opt convergence on gradients
        
        # save old gradient g -> dg and copy gnew to g
        dg = g.copy()
        g = gnew.copy()
        
        g_norm = np.sqrt(np.dot(g,g))
                
        # get max gnew component
        gtest = np.max(np.abs(g))
        if coord_type == "masswt":
            car_gtest = np.max(np.abs(g/inv_sqrt_mass))
            psi4.core.print_out("\n@@@ max g new component = %10.6f and norm = %10.6f  car_gtest = %10.6f"
                 % (gtest,g_norm,car_gtest))
        else:
            psi4.core.print_out("\n@@@ max g new component = %10.6f and norm = %10.6f" % (gtest,g_norm))
        
        if gtest < gtol:
            psi4.core.print_out("\n@Iter %3d - optimization converged on gtol - fini!!" % iter)
            g_conv = True
            #return fp,p,gnew
        # iter_record = [iter,energy,coords,grad,step,del_en,xtest,gtest,e_conv,x_conv,g_conv,hessin]
        
        iter_record = [iter,fp,p,gnew,xi,del_en,xtest,gtest,e_conv,x_conv,g_conv] # could add hessin?
        if coord_type == "masswt":
            iter_record.append(car_xtest)
            iter_record.append(car_gtest)
        dfpmin_iter_sum.append(iter_record)
            
        if e_conv and x_conv and g_conv:
            psi4.core.print_out("\n@@@@@@ BINGO: Energy, disp and grad all converged for %s using %s coords\n" % (mol.name(),coord_type))
            psi4.core.print_out("\n     >>>>>>>>>> Optimization summary - number of records %d <<<<<<<<<<<<<" % len(dfpmin_iter_sum))
            psi4.core.print_out("\n Iter       Energy          DEnergy  Econv   max_x  xconv   max_g  gconv ")

            for irec in range(len(dfpmin_iter_sum)):
                rec = dfpmin_iter_sum[irec]
                #print("%d: " % irec,rec)
                # check to see if rec[1] is an integer - due to doing more than 1 linsrch
                if type(rec[1]) == int:
                    # have line search info
                    psi4.core.print_out("\n %3d.%d %15.9f     linsrch alfa = %9.5f" % (rec[0],rec[1],rec[3],rec[2]))
                else:     # print regular dfpmin energy info
                    if coord_type == "masswt":
                        psi4.core.print_out("\n %3d   %15.9f %15.9f %i   %10.4e %i   %10.4e %i  cx %10.4e  cg %10.4e"
                            % (rec[0],rec[1],rec[5], rec[8], rec[6], rec[9], rec[7],rec[10],rec[11],rec[12]))
                    else:
                        psi4.core.print_out("\n %3d   %15.9f %15.9f %i   %10.4e %i   %10.4e %i" % 
                          (rec[0],rec[1],rec[5], rec[8], rec[6], rec[9], rec[7],rec[10]))
            psi4.core.print_out("\n      Optimized structure obtained from: %d energy and grad calcs" % totno_en_gr_evals)
            psi4.core.print_out("\n           >>>>>>>>>>> End-of-optimization sum <<<<<<<<<<<<<\n")
            return fp,p,gnew,hessin

        # now start doing the BFGS update for the inverse hessian hessin
        
        dg = g - dg
        hdg = np.dot(hessin,dg)
        
        # calc dot products for denomominators
        fac = np.dot(dg,xi)
        fae = np.dot(dg,hdg)
        sumdg = np.dot(dg,dg)
        sumxi = np.dot(xi,xi)
        
        if fac > np.sqrt(EPS*sumdg*sumxi):  # skip update if fac not sufficiently positive
            fac = 1.0/fac
            fad = 1.0/fae
            
            dg = fac* xi - fad*hdg
            
            #for i in range(nvar):
                #for j in range(i,nvar):
                    #hessin[i,j] += fac*xi[i]*xi[j] - fad*hdg[i]*hdg[j] + fae*dg[i]*dg[j]
                    #hessin[j,i] = hessin[i,j]
            
            for i in range(nvar):
                hessin[i,:] += fac*xi[i]*xi -fad*hdg[i]*hdg + fae*dg[i]*dg
                
            psi4.core.print_out("\n@Iter %3d hessin -->\n"% iter)
            psi4.core.print_out(str(hessin))
                    
        else:
            psi4.core.print_out("hessin not updated because")
            psi4.core.print_out("fac = %12.5e > sqrt(EPS*sumdg*sumxi) = %12.5e" % (fac,np.sqrt(EPS*sumdg*sumxi)))

        #
        # now get new step - already done
        
        #end of step loop - iter
    psi4.core.print_out(">>>>>>>>>>>>> Optimization quits because too many iterations")
    return fp,p,g,hessin

#################################### end of dfpmin ############################################
#


# In[6]:


#
#####################################################################################
#  first test of dfpmin by doing geom opt of co

#def jdh_optimizer(version,args=None):
def jdh_optimizer(version,mol=None,eg_opts='scf'):
    """ routine to call jdh bfgs optimization proceedure
        
        version = coordinate system: either "cart" or "masswt"
        
        initially args = (mol, eg_opts)
        
        when dfpmin called args needs to contain:
        the parameters to be passed to dftmin and ultimately the psi4 energy and gradient functions
         
        args = (mol, eg_opts, init_coords, init_com, inv_sqrt_mass, coord_type)
    where:
        mol = molecule class name
        eg_opts = energy/gradient options
        init_coords such that coords are zero 
        init_com = mol center of mass at init_coords
        inv_sqrt_mass = 1/sqrt(atmas[iat]) - used when coord_type = 'masswt'
        coord_type posibilities so far: 'cart'
        
        maxiter = no of geometry searches performed
        stpmx = max step size in individual geometry change
        etol = tolerance on desired energy change at min = fp(iter) - f(iter-1)
        gtol = tolerance on max gradient at converged minimum
        xtol = tolerance on max step size at converged minimum
        init_invhess = enables a guess at the starting invhess used in the optimization
        
     dfmin returns opt_code, fret, p_opt, gr_opt, opt_tol, invhess
     where: 
         opt_code = ?
         fret = energy value of optimized geom
         p_opt = optimized coordinates
         gr_opt = energy gradients at optimized geom
         opt_tol = [???] summarizing final opt tolerances
         invhess = final approximate invhess
    """
         
    
    # get parameters from args
    #(mol,eg_options) = args
    # do not use args to define (mol,eg_opts)
    # check for active molecule
    act_mol = psi4.core.get_active_molecule()
    if mol == act_mol:
        print("mol equals active mol:",mol)
    else:
        print("act mol = ",act_mol," whereas mol =",mol)
    if mol is None:
        mol = psi4.core.get_active_molecule()
    psi4.core.print_out("\n ============= Start of jdh_optimize of molecule %s ===================\n" % mol.name())
    psi4.core.print_out("jdh_opt_code module location: %s" % __file__)
    #print("\n=== Energy/gradient options: %s" % eg_opts)
    nat = mol.natom()
    psi4.core.print_out("\n Molecule %s contains %d atoms" % (mol.name(),nat))
    print_mol_coord_sum(mol,opt_stage="Initial")

    inv_sqrt_mass = np.ones(3*nat,dtype=float)

    in_iat = 0
    print("Atoms and initial coordinates in Molecule %s" % mol.name())
    for iat in range(nat):
        psi4.core.print_out("\n atom %d %3s %9.5f xyz = %12.7f  %12.7f  %12.7f" % (iat,mol.symbol(iat),mol.mass(iat),
              mol.xyz(iat)[0], mol.xyz(iat)[1], mol.xyz(iat)[2]))
        inv_mass = 1./np.sqrt(mol.mass(iat))
        for ii in range(3):
            inv_sqrt_mass[in_iat] = inv_mass
            in_iat += 1
    # set up init_com and get optimization method: "cart" or "masswt"

    #init_com = np.array(mol.center_of_mass())
    xcom = mol.center_of_mass()[0]
    ycom = mol.center_of_mass()[1]
    zcom = mol.center_of_mass()[2]
    #print("xcom,ycom,zcom: ",xcom,ycom,zcom)
    init_com = np.array([xcom,ycom,zcom],dtype=float)
    psi4.core.print_out("\n    Initial center of mass = %12.7f  %12.7f  %12.7f" % (init_com[0],init_com[1],init_com[2]))
    
    # get initial molecule coordinates
    print("\n Initial molecule coordinates in a.u. ==================")
    init_coords = mol.geometry().np
    print("++++ Molecule %s initial coordinates: ->\n" % mol.name(),init_coords)
    
    # set up displacement coordinates from init_coords
    p = np.zeros(nat*3,dtype=float)

    #  set up  masswt inv_hess if version == "masswt"
    if version == "masswt":
        inv_hess = np.identity(nat*3,dtype=float)
        ii = 0
        for iat in range(nat):
            for jj in range(3):
                inv_hess[ii,ii] = mol.mass(iat)
                ii += 1
        psi4.core.print_out("\n Formed mass weighted inv hessian for initial hessian")

    else:
        inv_hess = None


    print("====================== Start geometry optimization using %s coordinates" % version)

    # note init_coords is the starting molecule geom at beginning of optimization
    # the molecules optimized coords = init_coords + optp
    # init_coords is fixed and are cartesian coordinates --- optp can be simply cartesian or massweighted
    # at the starting geom --- optp is zero

    optfun, optp, optg, opt_hessin = dfpmin(en_gr_fun,p, args=(mol,eg_opts,init_coords,init_com,inv_sqrt_mass,version),
       init_invhess =  inv_hess)

    psi4.core.print_out("\n============ optimization completed ==============================\n")
    psi4.core.print_out("\nopt E = %20.10f" % optfun)
    psi4.core.print_out("\nopt disp coords = \n")
    psi4.core.print_out(str(optp))
    psi4.core.print_out("\ngradient at opt geom =\n")
    psi4.core.print_out(str(optg))
    psi4.core.print_out("\n==== approx inverse hessian ====\n")
    psi4.core.print_out(str(opt_hessin))
    psi4.core.print_out("\n======================== end of optimization =============================\n")
    #
    # get final coords
    if version == "cart":
        pass
    elif version == 'masswt':
        # coords are mass weighted - convert to cartessian
        optp = optp * inv_sqrt_mass # cartesian displacment

        # NEEDS WORK!!
        print("masswt disp coords as cart: ",optp)

    fin_coords = np.reshape(init_coords,(len(optp),)) + optp
    print("For %s coordinates - final cartesian coords are -->" % version, fin_coords)
    #print(" CO optdistance = %10.6f a.u." % (np.abs(fin_coords[2] - fin_coords[5])))
    
    # print summary of Optimized geometry
    ostage = "Opt " + version
    print_mol_coord_sum(mol,opt_stage=ostage)
    
    
    psi4.core.print_out("\n    Initial center of mass: %18.10f  %18.10f  %18.10f a.u."
             % (init_com[0],init_com[1],init_com[2]))

    xcom = mol.center_of_mass()[0]
    ycom = mol.center_of_mass()[1]
    zcom = mol.center_of_mass()[2]
    psi4.core.print_out("\nCOM for optimized molecule: %18.10f  %18.10f  %18.10f a.u.\n" % (xcom,ycom,zcom))
    
    psi4.core.print_out("\n===== Exit from jdh_optimize - return optimized energy =====\n")
    
    
    return optfun
    
##########################################################################################
#   end of jdh_optimizer code      24-dec-2019
##########################################################################################

#   set up main program to test optimizer code
if __name__ == "__main__":
    print("+----------------------------------------------------+")
    print("|     Start of main pgm to test jdh_opt_code.py      |")
    print("+----------------------------------------------------+")
    #
    # Memory specification
    psi4.set_memory(int(5e8))
    numpy_memory = 2

    # Set output file
    # TODO: originally dat --> out
    #psi4.core.set_output_file('output.out', False)

    # Define Physicist's water -- don't forget C1 symmetry!
    # mol = psi4.geometry("""
    # O
    # H 1 1.1
    # H 1 1.1 2 104
    # symmetry c1
    # """)


    ##########################################################################
    # start of setting up atomic coords, masses for different molecules
    # use_psi4_data = True if obtaining molecular info via psi4

    # in current program version - testing program with some simple molecules
    # set use_psi4_data = False
    #########################################################################

    #### check out using __file__

    print("check __file__ ",__file__)
    print("get __file__ abspath: ",os.path.abspath(__file__))

    # use_psi4_data = False  # set false when not using psi4 data
    use_psi4_data = True
    # (a) set up psi4 data
    # below is the psi4 setup
    if use_psi4_data:
        import psi4

        nthread = 4
        psi4.set_num_threads(nthread)

        # setup np print options
        print("get_print_options: ", np.get_printoptions())
        np.set_printoptions(precision=6, linewidth=120)

        # In[3]:


        # now read in psi4 wavefun file from a freq calculations
        # genmol_wfn.from_file("h1oac_nas_opt_wfn_freq.npy")

        #hess_file_name = "h1oac_nas_opt_wfn_freq.npy"
        #hess_file_name = "hbeh_r21beh_freq_wfn.npy"  # only has 2 real str modes
        #hess_file_name = "hbeh_r27beh_freq_wfn.npy"
        #hess_file_name = "co_opt_freq_wfn.npy"
        #hess_file_name = "hbeh_opt_hess_wfn.npy"
        #hess_file_name = "h2co_opt_hess_wfn.npy"
        #hess_file_name = "hooh_opt_hess_wfn.npy"
        hess_file_name = "h2co_opt_hess_wfn"
        file_wfn = psi4.core.Wavefunction.from_file(hess_file_name+'.npy')
        # Set output file
        psi4.core.set_output_file(hess_file_name+'.out', False)

        psi4.core.print_out("\n ============ Starting geom opt using %s.npy "
                            "data\n" % hess_file_name)
        print("=============== Start reading hessian matrix from file_wfn file")
        hess = file_wfn.hessian()

        # In[4]:

        file_mol = file_wfn.molecule()

        # print("file_mol geom = ",file_mol.geometry().np)

        mol_name = file_mol.name()
        num_file_at = file_mol.natom()
        file_geom = np.asarray(file_mol.geometry())
        print("no_ats in %s molecule = %d   file_geom.shape = " % (
            mol_name, num_file_at), file_geom.shape)

        npmass = np.asarray([file_mol.mass(iat) for iat in range(num_file_at)])

        print("\n=========================================================")

        print("  %s  --- Units = %s" % (
            file_wfn.molecule().name(), file_wfn.molecule().units()))
        print("            x            y             z       mass")

        #  included atom symbol (label) in print out
        at_label = []
        # set up a coordinate string
        opt_mol_geom_setup = ""
        for iat in range(num_file_at):
            at_label.append(file_wfn.molecule().label(iat))
            print("%3d %2s %12.7f %12.7f %12.7f %12.7f" % (iat, at_label[iat],
                                                           file_geom[iat, 0],
                                                           file_geom[iat, 1],
                                                           file_geom[iat, 2],
                                                           npmass[iat]))
            atom_str = "  %2s  %20.12f  %20.12f  %20.12f\n" % \
                       (at_label[iat], file_geom[iat,0],
                        file_geom[iat,1], file_geom[iat,2])
            opt_mol_geom_setup += atom_str
        opt_mol_geom_setup += "\n no_com \n no_reorient \n symmetry c1 \n " \
                            "units bohr"

        print("opt_mol_geom_setup -->\n",opt_mol_geom_setup)
        print("\n=========================================================")

        print("Psi4 %s center of mass = " % file_wfn.molecule().name(),
              file_wfn.molecule().center_of_mass())
        print("Psi4 %s rotational consts = " % file_wfn.molecule().name(),
              file_wfn.molecule().rotational_constants().np)
        print("and inertia tensor =>\n", file_wfn.molecule().inertia_tensor().np)
        print("Psi4 fixed com = %s   fixed orientation = %s" % (
        file_wfn.molecule().com_fixed(),
        file_wfn.molecule().orientation_fixed()))

        # In[8]:

        # get list of info available from wavefunction file
        #print("dir(file_wfn) -->\n",dir(file_wfn))
        # print("dir(genmol_wfn) -->\n",dir(genmol_wfn))
        #print("\n======== dir(file_wfn.molecule()) --> ",dir(file_wfn.molecule()))

        print(" Name of molecule = file_wfn.molecule.name()?",
              file_wfn.molecule().name())

        print("\nfile_wfn.basisset().name() = ", file_wfn.basisset().name())
        print("file_wfn.basisset().nbf() = ", file_wfn.basisset().nbf())
        print("file_wfn.nirrep() = ", file_wfn.nirrep())

        print("\nfile_wfn energy =", file_wfn.energy())

        # get list of info available from wavefunction file

        # print("\n======== type(file_wfn) --> ",type(file_wfn))
        # print("dir(file_wfn) -->\n",dir(file_wfn))

        # print("\n======== type(file_wfn.molecule()) -> ",type(file_wfn.molecule))
        # print("\n======== dir(file_wfn.molecule()) --> ",dir(file_wfn.molecule()))

        # hessian info
        # print("\n======== type(file_wfn.hessian()) -> ",type(file_wfn.hessian()))
        # print("\n======== dir(file_wfn.hessian()) -> ",dir(file_wfn.hessian()))


        # get some hessian info
        # print("type(hess) = ",type(hess))
        # print("dir(hess) -->",dir(hess))
        # print("==== (hess.__getattribute__(diagonalize) --> \n",hess.__getattribute__('diagonalize'))
        if hess.symmetry() == 0:
            print("hess.symmetry =", hess.symmetry())
            nphess = np.asarray(hess)
            print("\n file_wfn hess shape = ", hess.shape)
            # print out file_wfn.hessian
            #if debug_check:
                # print("file_wfn.hess --> numpy nphess  shape = ",nphess.shape, " -->\n",nphess)
                #hsf.print_hess(nphess,title="file_wfn hess matrix")
        else:
            print("+++++++++++++++ hess.symmetry() is not ZERO - need to worry about symmetry blocks in hessian")
            sys.exit("***** Program quits because hess.symmetry() is NOT ZERO")

        print("=========== End of working with numpy nphess <= file_wfn.hess ====")



        print("\n=========================================================")

        # set up opt_mol - separate class to molecule in hess file

        opt_mol = psi4.geometry(opt_mol_geom_setup)
        opt_mol.set_name(mol_name)

        # Computation options
        # psi4.set_options({'basis': 'aug-cc-pvdz',
        #                  'scf_type': 'df',
        # psi4.set_options({'basis': '6-31g',
        # check to see if optim converges in 1 step with aug-cc-pvdz basis
        psi4.set_options({'basis': 'aug-cc-pvdz',
                          'e_convergence': 1e-8,
                          'd_convergence': 1e-8})

        # probably show check energy type and list options later??

        # Get the SCF wavefunction & energies for H2O
        # scf_e0, scf_wfn = psi4.energy('scf', return_wfn=True)
        # print("A float and a Wavefunction object returned:", scf_e0, scf_wfn)

        # setup energy_gradient options
        eg_opts = 'scf'

        print("energy/gradient options: %s" % eg_opts)

        #  perform geometry optimization
        #
        # molecule class given by file_mol - which from above is:
        #file_mol = file_wfn.molecule()
        # version either "cart" or "masswt"
        # masswt either "trad" or "unit"
        version = "cart"

        # orig: opt_energy = jdh_optimizer(version, args=(co, "scf"))
        print("Going to optimize mol = %s " % opt_mol.name())
        opt_energy = jdh_optimizer(version, mol=opt_mol,eg_opts="scf")

        print("\n++++++++++++++ end of geom opt on %s  ++++++++++++++\n" 
               % opt_mol.name())
        # (0) calc frequencies from psi4 hessian wfn
        # set up and analyze traditional mass_wt hessian


        #   add in traditional ehess frequency analysis here

        print("\n++++ (0) Traditional atomic mass weighted freq calc using numerical diff ehess ++++\n")
        # second derivative matrix  nphess -> from above file_wfn read

        mwt_hess, umass, atmass_gmean, inv_hess, ret_freq_type, anal_freq, \
        anal_evec = hsa.hess_setup_anal(
            mol_name, at_label, npmass, file_geom, nphess,
            tran_rot_v=None,
            hess_type='ehess',
            approx_type=None,
            ehess_type='cart',
            mhess_type='trad',
            inv_hess=False,
            get_unproj_freq=False,
            get_proj_freq=True,
            anal_end_freq=True,
            prnt_mol_info=False)
        num_at = num_file_at
        mol_geom = file_geom
        units = file_mol.units()
        # print("  %s  --- Units = %s" % (file_wfn.molecule().name(),
        # file_wfn.molecule().units()))

        print('numerical frequencies - ret_freq_type = %s\n' % ret_freq_type,
              anal_freq)

        print("\n      ======= End of (0) %s frequencies from psi4 hess "
        "wavefn========\n\n" % mol_name)

    ####################################################################
    #  start of some simple molecules to test lindh approx hessian idea

    else:
        # case 0 - set up molecular data for H-Be-H
        mol_name = "H-Be-H"

        # Setup atoms: at_labels, coordinates(mol_geom) and their masses (npmass)
        at_label = ['H', 'Be', 'H']

        d = 2.1  # Be-H bondlength in atomic units (need to check)
        mol_geom = np.array(
            [[0., 0., -d], [0., 0., 0., ], [0., 0., d]], dtype = float)


        # orig  Be =4 huh? # npmass = np.array([1., 4., 1.], dtype=float)
        npmass = np.array([1., 9., 1.], dtype=float)
        num_at = len(npmass)
        units = "Angstrom"

        ############ end-of-case 0 ################

    print("\n++++++++++++++++++++++ Molecular data for %s ++++++++++++++++++++++"
          % mol_name)

    print("====================================================================\n")

    print("num_at in %s molecule = %d   mol_geom.shape = " %
          (mol_name, num_at),mol_geom.shape)


    print("\n=========================================================")

    #print("  %s  --- Units = %s" % (file_wfn.molecule().name(),
    # file_wfn.molecule().units()))
    print("  %s  --- Units = %s" % (mol_name, units))
    print("            x            y             z       mass")

    #  included atom symbol (label) in print out
    for iat in range(num_at):
        print("%3d %2s %12.7f %12.7f %12.7f %12.7f" % (iat, at_label[iat],
                                                       mol_geom[iat, 0], mol_geom[iat, 1], mol_geom[iat, 2], npmass[iat]))

    print("\n=========================================================")

    ####################################################################
    # -- examples of different calls to the hess_set_anal.py module -- #
    ####################################################################

    pass
