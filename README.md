# mean_field_cqed
Modified mean_field_cqed code for many-electron PIW model of a nanoparticle

TDHF::common_init() - main functions include
  Getting information about molecule, allocating appropriate memory for arrays (unmodified)
  Getting information about NP, allocating appropriate memory for arrays.  Modified to include information such as 
    Particle shape (spherical or cubic)
    Particle size (spherical -> R, cubic ->L)
    Number of orbitals (nS, specified by nmax = nxmax, nymax, nzmax for cubic, specified by nmax and lmax for spherical)
    Number of electrons (specified by user-supplied np_els)
    Number of occupied orbitals (np_occ = np_els/2)
    Number of virtual orbitals (np_virt = nS - np_occ)
   Note:  There is only 1 component of density matrix now - called Dre_plasmon and Dim_plasmon now, x,y,z, components 
   not mentioned anymore
   
   Q: Should Plasmon Coordinates be user-defined rather than hard-coded?
   
   Hp_x, Dip_x, Dip_y, Dip_z formed around line 336
   
   HInteraction formed around line 393
   
   Note:  Laser field needs to be handled... in particular, needs ability to read laser field from file and interpolate
   field values not explicitly represented in data file.  ExtField Function is on line ~625
   
   Q:  Dipole Potential Integrals - again, what does this do?
   
   Note:  Plasmon dipole moment calculated around line 986 - notice all components of dipole moment calculated with
   Dre_plasmon (no components in the density matrix, just the operator matrix).
   
   
