#include"psi4/psi4-dec.h"
#include"psi4/liboptions/liboptions.h"
//#include"psi4/libmints/mints.h"
#include"psi4/libmints/matrix.h"
#include"psi4/libmints/molecule.h"
#include"psi4/libmints/integral.h"
#include"psi4/libmints/mintshelper.h"
#include"psi4/libmints/basisset.h"
#include"psi4/libmints/petitelist.h"
#include"psi4/libmints/wavefunction.h"
#include"psi4/libpsio/psio.hpp"
#include"psi4/physconst.h"
#include"psi4/libqt/qt.h"
#include"psi4/psifiles.h"

#include"tdhf.h"
#include"blas.h"
#include"frozen_natural_orbitals.h"

#ifdef _OPENMP
    #include<omp.h>
#endif

using namespace psi;
using namespace fnocc;

namespace psi{ namespace tdhf_cqed {

std::shared_ptr<TDHF> MyTDHF;


TDHF::TDHF(std::shared_ptr<Wavefunction> reference_wavefunction,Options & options):
  Wavefunction(options)
{
    reference_wavefunction_ = reference_wavefunction;
    common_init();
}
TDHF::~TDHF(){
}
void TDHF::common_init(){

//    printf("Initializing ...\n");
    escf    = reference_wavefunction_->reference_energy();
    doccpi_ = reference_wavefunction_->doccpi();
    soccpi_ = reference_wavefunction_->soccpi();
    frzcpi_ = reference_wavefunction_->frzcpi();
    frzvpi_ = reference_wavefunction_->frzvpi();
    nmopi_  = reference_wavefunction_->nmopi();
    nsopi_  = reference_wavefunction_->nsopi();
    molecule_ = reference_wavefunction_->molecule();
    nirrep_ = reference_wavefunction_->nirrep();

//    printf("Getting coefficients ...\n");

    Da_ = SharedMatrix(reference_wavefunction_->Da());
    Ca_ = SharedMatrix(reference_wavefunction_->Ca());
    Fa_ = SharedMatrix(reference_wavefunction_->Fa());

//    printf("Getting orbital parameters ... %i	%i \n",nirrep_, nsopi_);

    epsilon_a_= std::shared_ptr<Vector>(new Vector(nirrep_, nsopi_));
//    printf("here1 ...\n");
    epsilon_a_->copy(reference_wavefunction_->epsilon_a().get());
//    printf("here2 ...\n");
    nalpha_ = reference_wavefunction_->nalpha();
//    printf("here3 ...\n");
    nbeta_  = reference_wavefunction_->nbeta();
    nso = nmo = ndocc = nvirt = nfzc = nfzv = 0;
//    printf("Assigning values ...\n");
    for (int h=0; h<nirrep_; h++){
        nfzc   += frzcpi_[h];
        nfzv   += frzvpi_[h];
        nso    += nsopi_[h];
        nmo    += nmopi_[h]-frzcpi_[h]-frzvpi_[h];
        ndocc  += doccpi_[h];
    }
//    printf("Exiting loop ...\n");
    ndoccact = ndocc - nfzc;
    nvirt    = nmo - ndoccact;

    if ( nfzc > 0 ) {
        throw PsiException("TDHF does not work with frozen core (yet).",__FILE__,__LINE__);
    }
    if ( nso != nmo ) {
        throw PsiException("TDHF does not work with nmo != nso (yet).",__FILE__,__LINE__);
    }
//    printf("Getting memory ...\n");

    // memory is from process::environment
    memory = Process::environment.get_memory();
    // set the wavefunction name
    name_ = "TDHF";

    // orbital energies
    eps = (double*)malloc(nmo*sizeof(double));
    memset((void*)eps,'\0',nmo*sizeof(double));
    int count=0;
    for (int h=0; h<nirrep_; h++){
        for (int norb = frzcpi_[h]; norb<doccpi_[h]; norb++){
            eps[count++] = epsilon_a_->get(h,norb);
        }
    }
    for (int h=0; h<nirrep_; h++){
        for (int norb = doccpi_[h]; norb<nmopi_[h]-frzvpi_[h]; norb++){
            eps[count++] = epsilon_a_->get(h,norb);
        //    printf("epsilon %i  = %lf \n",count, eps[count]);
        }
    }
    
    //printf("Computing the Kinetic and Potential energies ...\n");
    std::shared_ptr<MintsHelper> mints (new MintsHelper(reference_wavefunction_));
    T   = mints->so_kinetic();
    V   = mints->so_potential();

    SoToMo(Ca_->rowspi()[0],Ca_->colspi()[0],T->pointer(),Ca_->pointer());
    SoToMo(Ca_->rowspi()[0],Ca_->colspi()[0],V->pointer(),Ca_->pointer());

    // if freezing the core, need to add frozen core contributions to the one-electron integrals:
    //TransformIntegralsFull();
    TransformIntegrals();

    // testing 4-index integrals:
    tei = (double*)malloc(nmo*nmo*nmo*nmo*sizeof(double));
    memset((void*)tei,'\0',nmo*nmo*nmo*nmo*sizeof(double));
    F_DGEMM('n','t',nmo*nmo,nmo*nmo,nQ,2.0,Qmo,nmo*nmo,Qmo,nmo*nmo,0.0,tei,nmo*nmo);
    #pragma omp parallel for schedule (static)
    for (int p = 0; p < nmo; p++) {
        for (int q = 0; q < nmo; q++) {
            for (int r = 0; r < nmo; r++) {
                for (int s = 0; s < nmo; s++) {
                    //for (int Q = 0; Q < nQ; Q++) {
                        //tei[p*nmo*nmo*nmo+q*nmo*nmo+r*nmo+s] += 2.0 * Qmo[Q*nmo*nmo+p*nmo+q]*Qmo[Q*nmo*nmo+r*nmo+s];
                        //tei[p*nmo*nmo*nmo+q*nmo*nmo+r*nmo+s] -=       Qmo[Q*nmo*nmo+p*nmo+r]*Qmo[Q*nmo*nmo+s*nmo+q];
                    //}
                    tei[p*nmo*nmo*nmo+q*nmo*nmo+r*nmo+s] -=       C_DDOT(nQ,&Qmo[p*nmo+r],nmo*nmo,&Qmo[s*nmo+q],nmo*nmo);
                }
            }
        }
    }
 
    // Initialize Plasmon/Nanoparticle Variables
    n_scf_plasmon_states = options_.get_int("N_SCF_PLASMON_STATES");
    np_els               = options_.get_int("NP_ELECTRONS");
    np_occ = np_els / 2;
    
    if (options_.get_str("NP_SHAPE") == "CUBIC") {
      L = options_.get_double("NP_SIZE")/length_au;
      // nx_max = ny_max = nz_max = nmax
      nmax = options_.get_int("MAXIMUM_N");
      np_virt = nmax * nmax * nmax - np_occ;
    }
    else if (options_.get_str("NP_SHAPE") == "SPHERICAL") {
      R = options_.get_double("NP_SIZE")/length_au;
      nmax = options_.get_int("MAXIMUM_N");
      lmax = options_.get_int("MAXIMUM_L");
      np_virt = nmax * (lmax + 1) * (lmax + 1) - np_occ;
    }
    else {
      L = options_.get_double("NP_SIZE")/length_au;
      // nx_max = ny_max = nz_max = nmax
      nmax = options_.get_int("MAXIMUM_N");
      np_virt = nmax * nmax * nmax - np_occ;
    }
     
    nS = np_occ + np_virt;

    /*  char strings for component labels */
    comp_x = (char *)malloc(1000*sizeof(char));
    comp_y = (char *)malloc(1000*sizeof(char));
    comp_z = (char *)malloc(1000*sizeof(char));

    NPOrbE  = (int *)malloc(nS*sizeof(int));
    NPOrb_x = (int *)malloc(nS*sizeof(int));
    NPOrb_y = (int *)malloc(nS*sizeof(int));
    NPOrb_z = (int *)malloc(nS*sizeof(int));
    NPOrb_n = (int *)malloc(nS*sizeof(int));
    NPOrb_l = (int *)malloc(nS*sizeof(int));
    NPOrb_m = (int *)malloc(nS*sizeof(int));
 
    strcpy(comp_x,"X");
    strcpy(comp_y,"Y");
    strcpy(comp_z,"Z");


    int offset = 0;
    offset_dre           = offset; offset += nmo*nmo;
    offset_dim           = offset; offset += nmo*nmo;
    offset_dre_plasmon = offset; offset += nS*nS;
    offset_dim_plasmon = offset; offset += nS*nS;

    Dre        = (std::shared_ptr<Matrix>) (new Matrix(nmo,nmo));
    Dim        = (std::shared_ptr<Matrix>) (new Matrix(nmo,nmo));

    Dre_plasmon = (std::shared_ptr<Matrix>) (new Matrix(nS,nS));
    Dim_plasmon = (std::shared_ptr<Matrix>) (new Matrix(nS,nS));
       
    F1re       = (std::shared_ptr<Matrix>) (new Matrix(nmo,nmo));
    F1im       = (std::shared_ptr<Matrix>) (new Matrix(nmo,nmo));
    Fre        = (std::shared_ptr<Matrix>) (new Matrix(nmo,nmo));
    Fim        = (std::shared_ptr<Matrix>) (new Matrix(nmo,nmo));
    Fre_temp   = (std::shared_ptr<Matrix>) (new Matrix(nmo,nmo));
    Fim_temp   = (std::shared_ptr<Matrix>) (new Matrix(nmo,nmo));

    Dip_x              = (std::shared_ptr<Matrix>) (new Matrix(nS,nS));
    Dip_y              = (std::shared_ptr<Matrix>) (new Matrix(nS,nS));
    Dip_z              = (std::shared_ptr<Matrix>) (new Matrix(nS,nS));

    Hp_x               = (std::shared_ptr<Matrix>) (new Matrix(nS,nS));
    Hp_y               = (std::shared_ptr<Matrix>) (new Matrix(nS,nS));
    Hp_z               = (std::shared_ptr<Matrix>) (new Matrix(nS,nS));

    Hp_int_x           = (std::shared_ptr<Matrix>) (new Matrix(nS,nS));
    Hp_int_y           = (std::shared_ptr<Matrix>) (new Matrix(nS,nS));
    Hp_int_z           = (std::shared_ptr<Matrix>) (new Matrix(nS,nS));

    // TODO: these don't need to be class members.  They are only used here.
    // in fact, it would be better if the Hamiltonian manipulation was done as a 
    // separate function that could be called here.  This function has gotten
    // pretty unreadable.
    Hplasmon_total_x   = (std::shared_ptr<Matrix>) (new Matrix(nS,nS));
    Hplasmon_total_y   = (std::shared_ptr<Matrix>) (new Matrix(nS,nS));
    Hplasmon_total_z   = (std::shared_ptr<Matrix>) (new Matrix(nS,nS));

    Eigvec_x           = (std::shared_ptr<Matrix>) (new Matrix(nS,nS));
    Eigvec_y           = (std::shared_ptr<Matrix>) (new Matrix(nS,nS));
    Eigvec_z           = (std::shared_ptr<Matrix>) (new Matrix(nS,nS));

    htemp_x            = (std::shared_ptr<Matrix>) (new Matrix(nS,nS));

    Eigval_x           = (std::shared_ptr<Vector>) (new Vector(nS));
    Eigval_y           = (std::shared_ptr<Vector>) (new Vector(nS));
    Eigval_z           = (std::shared_ptr<Vector>) (new Vector(nS));

    temp_x = (std::shared_ptr<Matrix>) (new Matrix(nS,nS));
    temp_y = (std::shared_ptr<Matrix>) (new Matrix(nS,nS));
    temp_z = (std::shared_ptr<Matrix>) (new Matrix(nS,nS));

    // initial density matrices:
    // molecule
    for (int i = 0; i < ndoccact; i++) {
        Dre->pointer()[i][i] = 1.0;
    }
    // nanoparticle
    for (int i = 0; i < np_occ; i++) {
      Dre_plasmon->pointer()[i][i] = 1.0;
    }


    // get dipole integrals for molecule:
    dipole = mints->so_dipole();
    SoToMo(Ca_->rowspi()[0],Ca_->colspi()[0],dipole[0]->pointer(),Ca_->pointer());
    SoToMo(Ca_->rowspi()[0],Ca_->colspi()[0],dipole[1]->pointer(),Ca_->pointer());
    SoToMo(Ca_->rowspi()[0],Ca_->colspi()[0],dipole[2]->pointer(),Ca_->pointer());


    // get field polarization:
    polarization = (double*)malloc(sizeof(double)*3);
    memset((void*)polarization,'\0',3*sizeof(double));
    if (options_["POLARIZATION"].has_changed()){
       if (options_["POLARIZATION"].size() != 3)
          throw PsiException("The POLARIZATION array has the wrong dimensions",__FILE__,__LINE__);
       for (int i = 0; i < 3; i++) polarization[i] = options_["POLARIZATION"][i].to_double();
       int pol = 0;
       for (int i = 0; i < 3; i++) {
           if (fabs(polarization[i]) > 1e-9) {
               pol++;
           }
       }
     //  if (pol > 1) {
     //     throw PsiException("only plane polarized light supported",__FILE__,__LINE__);
     //  }
    }else{
       polarization[0] = 0.0;
       polarization[1] = 0.0;
       polarization[2] = 1.0;
    }

    // get plasmon coordinates:
    plasmon_coordinates = (double*)malloc(sizeof(double)*3);
    memset((void*)plasmon_coordinates,'\0',3*sizeof(double));
    if (options_["PLASMON_COORDINATES"].has_changed()){
       if (options_["PLASMON_COORDINATES"].size() != 3)
          throw PsiException("The PLASMON COORDINATES array has the wrong dimensions",__FILE__,__LINE__);
       for (int i = 0; i < 3; i++) plasmon_coordinates[i] = options_["PLASMON_COORDINATES"][i].to_double();
    }else{
       plasmon_coordinates[0] = 0.0;
       plasmon_coordinates[1] = 0.0;
       plasmon_coordinates[2] = 100.0;
    }

    // Getting molecular information to compute the center of mass
    std::shared_ptr<Molecule>mol=Process::environment.molecule();
    natom_ = mol->natom();
    com_ = (double*)malloc(sizeof(double)*3);
    memset((void*)com_,'\0',3*sizeof(double));
    
    double temp_x = 0.0;
    double temp_y = 0.0;
    double temp_z = 0.0;
    double temp_m = 0.0;
    for (int i = 0; i < natom_ ; i++){
       temp_x += mol->mass(i) * mol->x(i);         
       temp_y += mol->mass(i) * mol->y(i);         
       temp_z += mol->mass(i) * mol->z(i);         
       temp_m += mol->mass(i);
    }

    com_[0] = temp_x/temp_m;
    com_[1] = temp_y/temp_m;
    com_[2] = temp_z/temp_m;
  
    //printf(" %lf	%lf	%lf	\n",com_[0],com_[1],com_[2]);
    //exit(0);

    // nuclear contribution to dipole moment:
    nuc_dip_x_ = 0.0;
    nuc_dip_y_ = 0.0;
    nuc_dip_z_ = 0.0;
    for (int i = 0; i < natom_; i++) {
        nuc_dip_x_ += mol->Z(i) * mol->x(i);
        nuc_dip_y_ += mol->Z(i) * mol->y(i);
        nuc_dip_z_ += mol->Z(i) * mol->z(i);
    }

    nuc_dip = nuc_dip_x_ + nuc_dip_y_ + nuc_dip_z_ ;

    // Nanoparticle Hamiltonian
    Hp_x->zero();
    Hp_y->zero();
    Hp_z->zero();
    
    // Plasmonic dipole moment operator
    Dip_x->zero();
    Dip_y->zero();
    Dip_z->zero();
    for (int s=0; s<nS; s++){

      Hp_x->pointer()[s][s] = NP_h(s);

      for (int t=0; t<nS; t++) {

        double TDX = TDPEval(s, t, comp_x);
        double TDY = TDPEval(s, t, comp_y);
        double TDZ = TDPEval(s, t, comp_z);
     
        Dip_x->pointer()[s][t] = TDX;
        Dip_y->pointer()[s][t] = TDY;
        Dip_z->pointer()[s][t] = TDZ;
      }

    }

    Dip_x->print();
    Dip_y->print();
    Dip_z->print();
   
    // Plasmon Hamiltonian
    plasmon_e = (double*)malloc(sizeof(double)*3);
    memset((void*)plasmon_e,'\0',3*sizeof(double));
    if (options_["PLASMON_E"].has_changed()){
       if (options_["PLASMON_E"].size() != 3)
          throw PsiException("The PLASMON E array has the wrong dimensions",__FILE__,__LINE__);
       for (int i = 0; i < 3; i++) plasmon_e[i] = options_["PLASMON_E"][i].to_double();
    }else{
       plasmon_e[0] = 2.042/27.21138;  // Energy for the Au nanoparticle taken from Gray's paper
       plasmon_e[1] = 2.042/27.21138;
       plasmon_e[2] = 2.042/27.21138;
    }

    // Interaction Hamiltonian
    double r = (com_[0]-plasmon_coordinates[0])*(com_[0]-plasmon_coordinates[0])
             + (com_[1]-plasmon_coordinates[1])*(com_[1]-plasmon_coordinates[1])
             + (com_[2]-plasmon_coordinates[2])*(com_[2]-plasmon_coordinates[2]);
    r = sqrt(r);

    double delta_x = plasmon_coordinates[0] - com_[0];
    double delta_y = plasmon_coordinates[1] - com_[1];
    double delta_z = plasmon_coordinates[2] - com_[2];

    double * r_vector;

    r_vector = (double*)malloc(sizeof(double)*3);
    memset((void*)r_vector,'\0',3*sizeof(double));

    r_vector[0] = delta_x;
    r_vector[1] = delta_y;
    r_vector[2] = delta_z;

    double oer3 = 1.0 /(r*r*r);
    coupling_strength = 1.0 * oer3;

    // build interaction Hamiltonian (molecule -> plasmon) 
    HInteraction(&(Dre->pointer()[0][0]));
 
    // Diagonalize total plasmon Hamiltonian
    Hplasmon_total_x->copy(Hp_x);
    Hplasmon_total_x->add(Hp_int_x);

    Hplasmon_total_y->copy(Hp_y);
    Hplasmon_total_y->add(Hp_int_y);

    Hplasmon_total_z->copy(Hp_z);
    Hplasmon_total_z->add(Hp_int_z);

    double TINY = 1e-12;
    for (int i = 0; i < nS_scf; i++) {
        for (int j = 0; j < nS_scf; j++) {
            if ( fabs(Hplasmon_total_x->pointer()[i][j]) < TINY ) {
                Hplasmon_total_x->pointer()[i][j] = 0.0;
            }
            if ( fabs(Hplasmon_total_y->pointer()[i][j]) < TINY ) {
                Hplasmon_total_y->pointer()[i][j] = 0.0;
            }
            if ( fabs(Hplasmon_total_z->pointer()[i][j]) < TINY ) {
                Hplasmon_total_z->pointer()[i][j] = 0.0;
            }
        }
    }

    Hplasmon_total_x->diagonalize(Eigvec_x, Eigval_x);
    Hplasmon_total_y->diagonalize(Eigvec_y, Eigval_y);
    Hplasmon_total_z->diagonalize(Eigvec_z, Eigval_z);

    // transform Hamiltonians to new basis:
    PlasmonHamiltonianTransformation(Hp_x,Eigvec_x);
    PlasmonHamiltonianTransformation(Hp_y,Eigvec_y);
    PlasmonHamiltonianTransformation(Hp_z,Eigvec_z);
    PlasmonHamiltonianTransformation(Hp_int_x,Eigvec_x);
    PlasmonHamiltonianTransformation(Hp_int_y,Eigvec_y);
    PlasmonHamiltonianTransformation(Hp_int_z,Eigvec_z);
    PlasmonHamiltonianTransformation(Dip_x,Eigvec_x);
    PlasmonHamiltonianTransformation(Dip_y,Eigvec_y);
    PlasmonHamiltonianTransformation(Dip_z,Eigvec_z);
    
    total_time     = options_.get_double("TOTAL_TIME");
    time_step      = options_.get_double("TIME_STEP");
    laser_amp      = options_.get_double("LASER_AMP");
    laser_amp2     = options_.get_double("LASER_AMP2");
    laser_freq     = options_.get_double("LASER_FREQ");
    laser_freq2    = options_.get_double("LASER_FREQ2");
    delay_time     = options_.get_double("LASER_DELAY");
    transition_dpm = options_.get_double("LASER_TDPM");
    laser_time     = options_.get_double("LASER_TIME");
    total_iter     = total_time / time_step + 1;

    // which pulse shape do we want?
    if (options_.get_str("LASER_SHAPE") == "SIN_SQUARED") {
        // from prl:
        pulse_shape_ = 0;
    }else if (options_.get_str("LASER_SHAPE") == "TRAPEZOID") {
        // from 2007 schlegel paper (jcp 126, 244110 (2007))
        pulse_shape_ = 1;
    }else if (options_.get_str("LASER_SHAPE") == "PI_PULSE") {
        // pi pulse from licn paper
        pulse_shape_ = 2;
    }else if (options_.get_str("LASER_SHAPE") == "CONTINUOUS") {
        // continuous wave for rabi flopping
        pulse_shape_ = 3;   
    }else if (options_.get_str("LASER_SHAPE") == "GAUSSIAN"){
        // gaussian pulse
        pulse_shape_ = 4;
    }

    linear_response = false;

    // pad the correlation function with zeros just to get more output points
    //extra_pts = 4*(ttot/time_step+2);
    extra_pts = 0; //100000;//0;//1000000;
    // correlation function or dipole acceleration (fourier transformed)
    midpt = total_time/time_step+extra_pts + 1;
    corr_func = (fftw_complex*)fftw_malloc(sizeof(fftw_complex)*(int)(1+2*total_time/time_step+2+2*extra_pts));
    corr_func2 = (fftw_complex*)fftw_malloc(sizeof(fftw_complex)*(int)(1+2*total_time/time_step+2+2*extra_pts));
    corr_func3 = (fftw_complex*)fftw_malloc(sizeof(fftw_complex)*(int)(1+2*total_time/time_step+2+2*extra_pts));
    corr_func4 = (fftw_complex*)fftw_malloc(sizeof(fftw_complex)*(int)(1+2*total_time/time_step+2+2*extra_pts));
    corr_func5 = (fftw_complex*)fftw_malloc(sizeof(fftw_complex)*(int)(1+2*total_time/time_step+2+2*extra_pts));
    corr_func6 = (fftw_complex*)fftw_malloc(sizeof(fftw_complex)*(int)(1+2*total_time/time_step+2+2*extra_pts));
    corr_func7 = (fftw_complex*)fftw_malloc(sizeof(fftw_complex)*(int)(1+2*total_time/time_step+2+2*extra_pts));
    corr_func8 = (fftw_complex*)fftw_malloc(sizeof(fftw_complex)*(int)(1+2*total_time/time_step+2+2*extra_pts));
    corr_func9 = (fftw_complex*)fftw_malloc(sizeof(fftw_complex)*(int)(1+2*total_time/time_step+2+2*extra_pts));
  
    // maximum frequency to output (eV)
    max_freq = 30.0;

    // stencil for second derivative of dipole moment
    stencil = (double*)malloc(sizeof(double)*5);
    memset((void*)stencil,'\0',5*sizeof(double));

    // dipole potential integrals:
    DipolePotentialIntegrals();
}

void TDHF::DipolePotentialIntegrals() {

    std::shared_ptr<OneBodyAOInt> efp_ints(reference_wavefunction_->integral()->ao_efp_multipole_potential());

    int nbf = reference_wavefunction_->basisset()->nbf();
    int nao = reference_wavefunction_->basisset()->nao();

    std::vector< std::shared_ptr<Matrix> > mats;
    for(int i=0; i < 20; i++) {
        mats.push_back(std::shared_ptr<Matrix> (new Matrix(nao, nao)));
        mats[i]->zero();
    }

    // plasmon dipole potential felt by the molecule
    Vector3 coords(plasmon_coordinates[0],plasmon_coordinates[1],plasmon_coordinates[2]);
    efp_ints->set_origin(coords);
    efp_ints->compute(mats);

    // ao/so transformation
    std::shared_ptr<PetiteList> pet(new PetiteList(reference_wavefunction_->basisset(),reference_wavefunction_->integral(),true));
    std::shared_ptr<Matrix> U = pet->aotoso();

    std::shared_ptr<Matrix> Vx = Matrix::triplet(U,mats[1],U,true,false,false);
    std::shared_ptr<Matrix> Vy = Matrix::triplet(U,mats[2],U,true,false,false);
    std::shared_ptr<Matrix> Vz = Matrix::triplet(U,mats[3],U,true,false,false);

    // so/mo transformation
    SoToMo(Ca_->rowspi()[0],Ca_->colspi()[0],Vx->pointer(),Ca_->pointer());
    SoToMo(Ca_->rowspi()[0],Ca_->colspi()[0],Vy->pointer(),Ca_->pointer());
    SoToMo(Ca_->rowspi()[0],Ca_->colspi()[0],Vz->pointer(),Ca_->pointer());

    dipole_pot_x = (std::shared_ptr<Matrix>) (new Matrix(Vx));
    dipole_pot_y = (std::shared_ptr<Matrix>) (new Matrix(Vy));
    dipole_pot_z = (std::shared_ptr<Matrix>) (new Matrix(Vz));

}

// TODO: get rid of all of the extra containers to hold F
void TDHF::BuildFockThreeIndex(double * Dre_temp, double * Dim_temp, bool use_oe_terms) {

    Fre_temp->zero();
    Fim_temp->zero();

    // J
    F_DGEMV('t',nmo*nmo,nQ,2.0,Qmo,nmo*nmo,Dre_temp,1,0.0,Ire,1);
    F_DGEMV('t',nmo*nmo,nQ,2.0,Qmo,nmo*nmo,Dim_temp,1,0.0,Iim,1);

    F_DGEMV('n',nmo*nmo,nQ,0.5,Qmo,nmo*nmo,Ire,1,0.0,&(Fre_temp->pointer()[0][0]),1);
    F_DGEMV('n',nmo*nmo,nQ,0.5,Qmo,nmo*nmo,Iim,1,0.0,&(Fim_temp->pointer()[0][0]),1);

    if ( use_oe_terms ) {
        //Fre_temp->add(T);
        //Fre_temp->add(V);
        for (int i = 0; i < nmo; i++) {
            for (int j = 0; j < nmo; j++) {
                Fre_temp->pointer()[i][j] += T->pointer()[i+nfzc][j+nfzc];
                Fre_temp->pointer()[i][j] += V->pointer()[i+nfzc][j+nfzc];
            }
        }
    }
    
    // K(re)
    F_DGEMM('t','n',nmo,nmo*nQ,nmo,1.0,Dre_temp,nmo,Qmo,nmo,0.0,Ire,nmo);
    for (int q = 0; q < nQ; q++) {
        for (int a = 0; a < nmo; a++) {
            for (int j = 0; j < nmo; j++) {
                Iim[a*nmo+j] = Ire[q*nmo*nmo+j*nmo+a];
            }
        }
        C_DCOPY(nmo*nmo,Iim,1,Ire+q*nmo*nmo,1);
    }
    F_DGEMM('n','t',nmo,nmo,nQ*nmo,-1.0 * 0.5,Ire,nmo,Qmo,nmo,1.0,&(Fre_temp->pointer()[0][0]),nmo);
    //F_DGEMM('n','t',nmo,nmo,nQ*nmo,-1.0,Ire,nmo,Qmo,nmo,1.0,&(Fre_temp->pointer()[0][0]),nmo);
    // K(im)
    F_DGEMM('t','n',nmo,nmo*nQ,nmo,1.0,Dim_temp,nmo,Qmo,nmo,0.0,Ire,nmo);
    for (int q = 0; q < nQ; q++) {
        for (int a = 0; a < nmo; a++) {
            for (int j = 0; j < nmo; j++) {
                Iim[a*nmo+j] = Ire[q*nmo*nmo+j*nmo+a];
            }
        }
        C_DCOPY(nmo*nmo,Iim,1,Ire+q*nmo*nmo,1);
    }
    F_DGEMM('n','t',nmo,nmo,nQ*nmo,-1.0 * 0.5,Ire,nmo,Qmo,nmo,1.0,&(Fim_temp->pointer()[0][0]),nmo);
    //F_DGEMM('n','t',nmo,nmo,nQ*nmo,-1.0,Ire,nmo,Qmo,nmo,1.0,&(Fim_temp->pointer()[0][0]),nmo);

    if ( use_oe_terms ) {
        for (int i = 0; i < nmo; i++) {
            for (int j = 0; j < nmo; j++) {
                Fre_temp->pointer()[i][j] -= ext_field * dipole[0]->pointer()[i+nfzc][j+nfzc] * polarization[0];
                Fre_temp->pointer()[i][j] -= ext_field * dipole[1]->pointer()[i+nfzc][j+nfzc] * polarization[1];
                Fre_temp->pointer()[i][j] -= ext_field * dipole[2]->pointer()[i+nfzc][j+nfzc] * polarization[2];
            }
        }
    }
}


// TODO: get rid of all of the extra containers to hold F
// Fij = sum_kl Dkl ( 2 (ij|kl) - (il|kj) )
// Fij = sum_kl Dkl t(kl|ij)
void TDHF::BuildFock(double * Dre_temp, double * Dim_temp, bool use_oe_terms) {

    Fre_temp->zero();
    Fim_temp->zero();

    // TODO: out-of-core version
    F_DGEMV('t',nmo*nmo,nmo*nmo,1.0,tei,nmo*nmo,Dre_temp,1,0.0,&(Fre_temp->pointer()[0][0]),1);
    F_DGEMV('t',nmo*nmo,nmo*nmo,1.0,tei,nmo*nmo,Dim_temp,1,0.0,&(Fim_temp->pointer()[0][0]),1);

    if ( use_oe_terms ) {
        //Fre_temp->add(T);
        //Fre_temp->add(V);
        for (int i = 0; i < nmo; i++) {
            for (int j = 0; j < nmo; j++) {
                Fre_temp->pointer()[i][j] += T->pointer()[i+nfzc][j+nfzc];
                Fre_temp->pointer()[i][j] += V->pointer()[i+nfzc][j+nfzc];
            }
        }
    }

    if ( use_oe_terms ) {
        for (int i = 0; i < nmo; i++) {
            for (int j = 0; j < nmo; j++) {
                Fre_temp->pointer()[i][j] -= ext_field * dipole[0]->pointer()[i+nfzc][j+nfzc] * polarization[0];
                Fre_temp->pointer()[i][j] -= ext_field * dipole[1]->pointer()[i+nfzc][j+nfzc] * polarization[1];
                Fre_temp->pointer()[i][j] -= ext_field * dipole[2]->pointer()[i+nfzc][j+nfzc] * polarization[2];
            }
        }
    }

}

void TDHF::ExtField(double curtime){ 

    //Vext->zero();
    double sigma = laser_time*0.5;
    if (!linear_response) {

        // add external field

        ext_field = 0.0;

        if (pulse_shape_ == 0 ) {

            // from prl:
            if ( curtime < laser_time ) {
                ext_field = sin(M_PI*curtime/(laser_time));
                ext_field *= ext_field*laser_amp*sin(laser_freq*curtime);
            }

        } else if ( pulse_shape_ == 1 ) {

            // from 2007 schlegel paper (jcp 126, 244110 (2007))
            if (curtime <= 2.0 * M_PI / laser_freq)      ext_field = laser_freq * curtime / (2.0 * M_PI) * laser_amp;
            else if (curtime <= 4.0 * M_PI / laser_freq) ext_field = laser_amp;
            else if (curtime <= 6.0 * M_PI / laser_freq) ext_field = (3.0 - laser_freq * curtime / (2.0 * M_PI) ) * laser_amp;
            ext_field *= sin(laser_freq*curtime);

        } else if ( pulse_shape_ == 2 ) {

            // pi pulse from licn paper
            double sigma = laser_time*0.5;
            if ( curtime < laser_time ) {
                ext_field = cos(M_PI*curtime/(2.0*sigma) + 0.5*M_PI);
                ext_field *= M_PI/(sigma*transition_dpm) * ext_field * laser_amp * cos(laser_freq*curtime);
            }

        } else if ( pulse_shape_ == 3 ) {

            // continuous wave for rabi flopping
            ext_field = laser_amp*sin(laser_freq*curtime); 
        } else if ( pulse_shape_ == 4 ) {

            // Gaussian pulse
            ext_field  = laser_amp * exp(-((curtime-1.5*laser_time)*(curtime-1.5*laser_time))
                       / (0.3606738*laser_time*laser_time)) * sin(laser_freq*curtime);
            ext_field += laser_amp2 * exp(-((curtime-delay_time - 1.5*laser_time)*(curtime-delay_time - 1.5*laser_time))
                       / (0.3606738*laser_time*laser_time)) * sin(laser_freq2*curtime);

        }
    }
}

void TDHF::PlasmonHamiltonianTransformation(std::shared_ptr<Matrix> Ham,std::shared_ptr<Matrix>Eigvec) {
    std::shared_ptr<Matrix> temp (new Matrix(Ham));
    temp->zero();
    /*for (int A=0; A < nS_scf; A++){
        for (int B=0; B < nS_scf; B++){
            for (int C=0; C < nS_scf; C++){
                temp->pointer()[A][B] += Ham->pointer()[A][C]*Eigvec->pointer()[C][B];
            }
        }
    }

    F_DGEMM('n','n',nS_scf,nS_scf,nS_scf,1.0,&(Eigvec->pointer()[0][0]),nS_scf,

    Ham->zero();

    for (int A=0; A < nS_scf; A++){
        for (int B=0; B < nS_scf; B++){
            for (int C=0; C < nS_scf; C++){
                Ham->pointer()[A][B] += Eigvec->pointer()[C][A]*temp->pointer()[C][B];
            }
        }
    }*/

    F_DGEMM('n','n',nS_scf,nS_scf,nS_scf,1.0,&(Eigvec->pointer()[0][0]),nS_scf,&(Ham->pointer()[0][0]),nS_scf,0.0,&(htemp_x->pointer()[0][0]),nS_scf);
    F_DGEMM('n','t',nS_scf,nS_scf,nS_scf,1.0,&(htemp_x->pointer()[0][0]),nS_scf,&(Eigvec->pointer()[0][0]),nS_scf,0.0,&(Ham->pointer()[0][0]),nS_scf);
}

void TDHF::HInteraction(double * D1) {

    double r = (com_[0]-plasmon_coordinates[0])*(com_[0]-plasmon_coordinates[0])
             + (com_[1]-plasmon_coordinates[1])*(com_[1]-plasmon_coordinates[1])
             + (com_[2]-plasmon_coordinates[2])*(com_[2]-plasmon_coordinates[2]);
    r = sqrt(r);

    double delta_x = plasmon_coordinates[0] - com_[0];
    double delta_y = plasmon_coordinates[1] - com_[1];
    double delta_z = plasmon_coordinates[2] - com_[2];

    double * r_vector;

    r_vector = (double*)malloc(sizeof(double)*3);
    memset((void*)r_vector,'\0',3*sizeof(double));

    r_vector[0] = delta_x;
    r_vector[1] = delta_y;
    r_vector[2] = delta_z;

    e_dip_x = 2.0*C_DDOT(nso*nso,D1,1,&(dipole[0]->pointer())[0][0],1);
    e_dip_y = 2.0*C_DDOT(nso*nso,D1,1,&(dipole[1]->pointer())[0][0],1);
    e_dip_z = 2.0*C_DDOT(nso*nso,D1,1,&(dipole[2]->pointer())[0][0],1);

    memset((void*)&(Hp_int_x->pointer()[0][0]),'\0',nS_scf*nS_scf*sizeof(double));
    memset((void*)&(Hp_int_y->pointer()[0][0]),'\0',nS_scf*nS_scf*sizeof(double));
    memset((void*)&(Hp_int_z->pointer()[0][0]),'\0',nS_scf*nS_scf*sizeof(double));

    double tx, ty, tz;
    for (int A = 0; A < nS; A++) {
      for (int B = 0; B < nS; B++) {
     
        tx = (e_dip_x + nuc_dip_x_)*Dip_x->pointer()[A][B];
        tx -= 3.0*Dip_x->pointer()[A][B]*delta_x*(r_vector[0]*(e_dip_x + nuc_dip_x_) + r_vector[1]*(e_dip_y + nuc_dip_y_) + r_vector[2]*(e_dip_z + nuc_dip_z_))/(r*r);
        tx *= coupling_strength;

        ty = (e_dip_y + nuc_dip_y_)*Dip_y->pointer()[A][B];
        ty -= 3.0*Dip_y->pointer()[A][B]*delta_y*(r_vector[0]*(e_dip_x + nuc_dip_x_) + r_vector[1]*(e_dip_y + nuc_dip_y_) + r_vector[2]*(e_dip_z + nuc_dip_z_))/(r*r);
        ty *= coupling_strength; 

        tz = (e_dip_z + nuc_dip_z_)*Dip_z->pointer()[A][B];
        tz -= 3.0*Dip_z->pointer()[A][B]*delta_z*(r_vector[0]*(e_dip_x + nuc_dip_x_) + r_vector[1]*(e_dip_y + nuc_dip_y_) + r_vector[2]*(e_dip_z + nuc_dip_z_))/(r*r);
        tz *= coupling_strength;
            
        Hp_int_x->pointer()[A][B] = tx;
        Hp_int_y->pointer()[A][B] = ty;
        Hp_int_z->pointer()[A][B] = tz;
        }
    }

    free(r_vector);
}

// JJFNote: Need to clarify difference between temp and kre - temp = current density matrix vs k is density matrix graident
void TDHF::InteractionContribution(double * tempr,
                                   double * tempi,
                                   double * kre,
                                   double * kim, 
                                   double * tempr_p,
                                   double * tempi_p,
                                   double * kre_p,
                                   double * kim_p, 
                                   std::shared_ptr<Matrix> Hp_int,
                                   std::shared_ptr<Matrix> dipole_pot,
                                   double mdip, 
                                   double pdip) {


    // This gets the molecule->nanoparticle interaction term 
    // (requires molecule dipole expectation value, nanoparticle dipole matrix) 
    HInteraction(tempr);
    PlasmonHamiltonianTransformation(Hp_int_x,Eigvec_x);
    PlasmonHamiltonianTransformation(Hp_int_y,Eigvec_y);
    PlasmonHamiltonianTransformation(Hp_int_z,Eigvec_z);

    // contribution to plasmon:

    //for (int A=0; A<nS; A++){
    //    for (int B=0; B<nS; B++){
    //        double dumr = 0.0;
    //        double dumi = 0.0;
    //        for (int C=0; C<nS; C++){

    //             // remember, only excite one mode for now
    //             kre_p[A*nS+B] -= Hp_int->pointer()[A][C]*tempi_p[C*nS+B];    
    //             kre_p[A*nS+B] += Hp_int->pointer()[C][B]*tempi_p[A*nS+C];    
    //      
    //             kim_p[A*nS+B] += Hp_int->pointer()[A][C]*tempr_p[C*nS+B];    
    //             kim_p[A*nS+B] -= Hp_int->pointer()[C][B]*tempr_p[A*nS+C];    
    //        }
    //    }
    //}
    F_DGEMM('n','n',nS,nS,nS,-1.0,tempi_p,nS,&(Hp_int->pointer()[0][0]),nS_scf,1.0,kre_p,nS);
    F_DGEMM('n','n',nS,nS,nS,1.0,&(Hp_int->pointer()[0][0]),nS_scf,tempi_p,nS,1.0,kre_p,nS);

    F_DGEMM('n','n',nS,nS,nS,1.0,tempr_p,nS,&(Hp_int->pointer()[0][0]),nS_scf,1.0,kim_p,nS);
    F_DGEMM('n','n',nS,nS,nS,-1.0,&(Hp_int->pointer()[0][0]),nS_scf,tempr_p,nS,1.0,kim_p,nS);

    // contribution to electron
    //for (int i=0; i<nmo; i++){
    //    for (int j=0; j<nmo; j++){

    //        double dumr = 0.0;
    //        double dumi = 0.0;
    //        for (int q = 0; q < nmo; q++) {
    //            dumr += tempr[i*nmo+q] * dipole_pot->pointer()[j+nfzc][q+nfzc];
    //            dumr -= tempr[q*nmo+j] * dipole_pot->pointer()[q+nfzc][i+nfzc];

    //            dumi += tempi[i*nmo+q] * dipole_pot->pointer()[j+nfzc][q+nfzc];
    //            dumi -= tempi[q*nmo+j] * dipole_pot->pointer()[q+nfzc][i+nfzc];
    //        }

    //        kre[i*nmo+j] += dumi * pdip * (-1);
    //        kim[i*nmo+j] -= dumr * pdip * (-1);

    //    }
    //}

    F_DGEMM('t','n',nmo,nmo,nmo, pdip * (-1),&(dipole_pot->pointer()[0][0]),nmo+nfzc,tempi,nmo,1.0,kre,nmo);
    F_DGEMM('n','t',nmo,nmo,nmo,-pdip * (-1),tempi,nmo,&(dipole_pot->pointer()[0][0]),nmo+nfzc,1.0,kre,nmo);

    F_DGEMM('t','n',nmo,nmo,nmo,-pdip * (-1),&(dipole_pot->pointer()[0][0]),nmo+nfzc,tempr,nmo,1.0,kim,nmo);
    F_DGEMM('n','t',nmo,nmo,nmo, pdip * (-1),tempr,nmo,&(dipole_pot->pointer()[0][0]),nmo+nfzc,1.0,kim,nmo);
}


void TDHF::PlasmonContribution(double * tempr,
                               double * tempi,
                               double * kre,
                               double * kim, 
                               std::shared_ptr<Matrix> dip, 
                               std::shared_ptr<Matrix> Ham, 
                               double pol) {


    // one-particle part of uncoupled plasmon hamiltonian: 

    //for (int A=0; A<nS; A++){
    //    for (int B=0; B<nS; B++){
    //        for (int C=0; C<nS; C++){
    //             kre[A*nS+B] -= Ham->pointer()[A][C]*tempi[C*nS+B];    
    //             kre[A*nS+B] += Ham->pointer()[C][B]*tempi[A*nS+C];    
    //      
    //             kim[A*nS+B] += Ham->pointer()[A][C]*tempr[C*nS+B];
    //             kim[A*nS+B] -= Ham->pointer()[C][B]*tempr[A*nS+C];
    //         }
    //    }
    //}
    F_DGEMM('n','n',nS,nS,nS,-1.0,tempi,nS,&(Ham->pointer()[0][0]),nS_scf,1.0,kre,nS);
    F_DGEMM('n','n',nS,nS,nS,1.0,&(Ham->pointer()[0][0]),nS_scf,tempi,nS,1.0,kre,nS);

    F_DGEMM('n','n',nS,nS,nS,1.0,tempr,nS,&(Ham->pointer()[0][0]),nS_scf,1.0,kim,nS);
    F_DGEMM('n','n',nS,nS,nS,-1.0,&(Ham->pointer()[0][0]),nS_scf,tempr,nS,1.0,kim,nS);

    // external field part: 

    //for (int A=0; A<nS; A++){
    //    for (int B=0; B<nS; B++){
    //        for (int C=0; C<nS; C++){
    //            kre[A*nS+B] -= dip->pointer()[A][C]*tempi[C*nS+B]*(-ext_field)*pol;    
    //            kre[A*nS+B] += dip->pointer()[C][B]*tempi[A*nS+C]*(-ext_field)*pol;    
    //       
    //            kim[A*nS+B] += dip->pointer()[A][C]*tempr[C*nS+B]*(-ext_field)*pol;    
    //            kim[A*nS+B] -= dip->pointer()[C][B]*tempr[A*nS+C]*(-ext_field)*pol;    
    //        }
    //    }
    //}
    F_DGEMM('n','n',nS,nS,nS,-1.0*(-ext_field)*pol,tempi,nS,&(dip->pointer()[0][0]),nS_scf,1.0,kre,nS);
    F_DGEMM('n','n',nS,nS,nS,1.0*(-ext_field)*pol,&(dip->pointer()[0][0]),nS_scf,tempi,nS,1.0,kre,nS);

    F_DGEMM('n','n',nS,nS,nS,1.0*(-ext_field)*pol,tempr,nS,&(dip->pointer()[0][0]),nS_scf,1.0,kim,nS);
    F_DGEMM('n','n',nS,nS,nS,-1.0*(-ext_field)*pol,&(dip->pointer()[0][0]),nS_scf,tempr,nS,1.0,kim,nS);

}

void TDHF::BuildLindblad(double * tempr,
                         double * tempi,
                         double * kre,
                         double * kim) {

    plasmon_dr = options_.get_double("PLASMON_DR");
    for (int s = 0; s < nS; s++){
        for (int p = 0; p < nS; p++){
            kre[s*nS+p] -= 0.5*plasmon_dr*tempr[s*nS+p]*(s+p);
            kim[s*nS+p] -= 0.5*plasmon_dr*tempi[s*nS+p]*(s+p);
            if (s < nS-1 && p < nS-1){
               kre[s*nS+p] += plasmon_dr*tempr[(s+1)*nS+(p+1)]*sqrt((s+1)*(p+1)); 
               kim[s*nS+p] += plasmon_dr*tempi[(s+1)*nS+(p+1)]*sqrt((s+1)*(p+1)); 
            }
        }
    }
}

// so->mo transformation for 1-body matrix
void TDHF::SoToMo(int nsotemp,int nmotemp,double**mat,double**trans){
  double*tmp = (double*)malloc(sizeof(double)*nsotemp*nsotemp);
  memset((void*)tmp,'\0',nsotemp*nsotemp*sizeof(double));
  F_DGEMM('n','n',nmotemp,nsotemp,nsotemp,1.0,&trans[0][0],nmotemp,&mat[0][0],nsotemp,0.0,&tmp[0],nmotemp);
  F_DGEMM('n','t',nmotemp,nmotemp,nsotemp,1.0,&tmp[0],nmotemp,&trans[0][0],nmotemp,0.0,&mat[0][0],nsotemp);
  free(tmp);
}

double TDHF::compute_energy() {

    // rk4 -> just 2*(nmo*nmo+nS*nS) not 2*(nmo*nmo+3*nS*nS)
    long int rk4_dim = 2 * (nmo*nmo+nS*nS);
    double * rk4_soln = (double*)malloc( rk4_dim * sizeof(double) );
    double * rk4_temp = (double*)malloc( rk4_dim * sizeof(double) );
    double * rk4_k1   = (double*)malloc( rk4_dim * sizeof(double) );
    double * rk4_k2   = (double*)malloc( rk4_dim * sizeof(double) );
    double * rk4_k3   = (double*)malloc( rk4_dim * sizeof(double) );
    double * rk4_k4   = (double*)malloc( rk4_dim * sizeof(double) );

    fftw_iter   = 0;

    for ( int iter = 0; iter < total_iter; iter++ ) {

        // RK4
        // y(n+1) = y( n ) + 1/6 h ( k1 + 2k2 + 2k3 + k4 )
        // t(n+1) = t( n ) + h

        // rk4 solution buffer should contain: Dre, Dim, Dpxre, Dpxim, Dpyre, Dpyim, Dpzre, Dpzim
         
        for (int i = 0; i < nmo; i++) {
            for (int j = 0; j < nmo; j++) {
                rk4_soln[offset_dre + i*nmo+j] = Dre->pointer()[i][j];
                rk4_soln[offset_dim + i*nmo+j] = Dim->pointer()[i][j];
            }
        }

        for (int i = 0; i < nS; i++) {
            for (int j = 0; j < nS; j++) {
                rk4_soln[offset_dre_plasmon + i*nS+j] = Dre_plasmon->pointer()[i][j];
                rk4_soln[offset_dim_plasmon + i*nS+j] = Dim_plasmon->pointer()[i][j];
            }
        }

        // RK4
        // y(n+1) = y( n ) + 1/6 h ( k1 + 2k2 + 2k3 + k4 )
        // t(n+1) = t( n ) + h

        memset((void*)rk4_k1,'\0',rk4_dim*sizeof(double));;
        RK4(rk4_dim, rk4_soln, rk4_k1, rk4_k1, rk4_temp, iter*time_step + 0.0 * time_step, 0.0);
        RK4(rk4_dim, rk4_soln, rk4_k2, rk4_k1, rk4_temp, iter*time_step + 0.5 * time_step, 0.5);
        RK4(rk4_dim, rk4_soln, rk4_k3, rk4_k2, rk4_temp, iter*time_step + 0.5 * time_step, 0.5);
        RK4(rk4_dim, rk4_soln, rk4_k4, rk4_k3, rk4_temp, iter*time_step + 1.0 * time_step, 1.0);

        // y(n+1) = y( n ) + 1/6 h ( k1 + 2k2 + 2k3 + k4 )

        C_DAXPY(rk4_dim,1.0/6.0 * time_step,rk4_k1,1,rk4_soln,1);
        C_DAXPY(rk4_dim,2.0/6.0 * time_step,rk4_k2,1,rk4_soln,1);
        C_DAXPY(rk4_dim,2.0/6.0 * time_step,rk4_k3,1,rk4_soln,1);
        C_DAXPY(rk4_dim,1.0/6.0 * time_step,rk4_k4,1,rk4_soln,1);

        for (int i = 0; i < nmo; i++) {
            for (int j = 0; j < nmo; j++) {
                Dre->pointer()[i][j] = rk4_soln[offset_dre + i*nmo+j];
                Dim->pointer()[i][j] = rk4_soln[offset_dim + i*nmo+j];
            }
        }
        for (int i = 0; i < nS; i++) {
            for (int j = 0; j < nS; j++) {
                Dre_plasmon->pointer()[i][j] = rk4_soln[offset_dre_plasmon + i*nS+j];
                Dim_plasmon->pointer()[i][j] = rk4_soln[offset_dim_plasmon + i*nS+j];
            }
        }

        // evaluate dipole moment:

        e_dip_x = 2.0*C_DDOT(nso*nso,&(Dre->pointer())[0][0],1,&(dipole[0]->pointer())[0][0],1);
        e_dip_y = 2.0*C_DDOT(nso*nso,&(Dre->pointer())[0][0],1,&(dipole[1]->pointer())[0][0],1);
        e_dip_z = 2.0*C_DDOT(nso*nso,&(Dre->pointer())[0][0],1,&(dipole[2]->pointer())[0][0],1);

        dipole_p_x = 0.0;
        dipole_p_y = 0.0;
        dipole_p_z = 0.0;

        // Using loops instead of DDOT
        
        temp_x->zero();
        temp_y->zero();
        temp_z->zero();

        //  Only x-componet of plasmon density matrix
        //  temp_xi = D*mu_xi
        for (int A=0; A<nS; A++){
            for (int B=0; B<nS; B++){
                for (int C=0; C<nS; C++){
                    temp_x->pointer()[A][B] += Dre_plasmon->pointer()[A][C]*Dip_x->pointer()[C][B];
                    temp_y->pointer()[A][B] += Dre_plasmon->pointer()[A][C]*Dip_y->pointer()[C][B];
                    temp_z->pointer()[A][B] += Dre_plasmon->pointer()[A][C]*Dip_z->pointer()[C][B];
                }
            }
        }

        // pop = trace(D)
        double pop = 0.0;
        for (int A=0; A<nS; A++){
            pop += Dre_plasmon->pointer()[A][A]*A;
        }
  
        //  dipole_p_xi = trace(D*mu_xi) = trace(temp_xi)
        for (int A=0; A<nS; A++){
            dipole_p_x += temp_x->pointer()[A][A];
            dipole_p_y += temp_y->pointer()[A][A];
            dipole_p_z += temp_z->pointer()[A][A];
        } 

        // start accumulating the dipole acceleration after the pulse is over
        //if (!linear_response && iter * time_step >2000 && iter *time_step < 2000+1.0 / laser_freq * 2.0 * M_PI)
        //if (!linear_response && iter * time_step >laser_time){
           corr_func[fftw_iter][0]  = e_dip_x;
           corr_func[fftw_iter][1]  = 0.0;

           corr_func2[fftw_iter][0] = e_dip_y;
           corr_func2[fftw_iter][1] = 0.0;

           corr_func3[fftw_iter][0] = e_dip_z;
           corr_func3[fftw_iter][1] = 0.0;
          
           corr_func4[fftw_iter][0] = dipole_p_x;
           corr_func4[fftw_iter][1] = 0.0;
       
           corr_func5[fftw_iter][0] = dipole_p_y;
           corr_func5[fftw_iter][1] = 0.0;
       
           corr_func6[fftw_iter][0] = dipole_p_z;
           corr_func6[fftw_iter][1] = 0.0;

           ExtField(iter*time_step);

           corr_func7[fftw_iter][0] = ext_field * polarization[0];
           corr_func7[fftw_iter][1] = 0.0;
       
           corr_func8[fftw_iter][0] = ext_field * polarization[1];
           corr_func8[fftw_iter][1] = 0.0;
       
           corr_func9[fftw_iter][0] = ext_field * polarization[2];
           corr_func9[fftw_iter][1] = 0.0;
       
           fftw_iter++;
        //}

        //CorrelationFunction();

        double en = Process::environment.molecule()->nuclear_repulsion_energy();
        
        outfile->Printf("@POP %20.12le   %20.12le \n",iter*time_step,pop);
        outfile->Printf("@TIME %20.12le %20.12le %20.12le %20.12le %20.12le %20.12le %20.12le %20.12le %20.12le %20.12le\n",iter*time_step,
            e_dip_x,e_dip_y,e_dip_z,
            dipole_p_x,dipole_p_y,dipole_p_z,
            ext_field * polarization[0],ext_field * polarization[1],ext_field * polarization[2]);

    }

    // fourier transform and spectrum
    FFTW();
    Spectrum();

    free(corr_func);
    free(corr_func2);
    free(corr_func3);
    free(corr_func4);
    free(corr_func5);
    free(corr_func6);
    free(corr_func7);
    free(corr_func8);
    free(corr_func9);
    

    return 0.0;
}

void TDHF::RK4( int rk4_dim, double * rk4_soln , double * rk4_out, double * rk4_in , double * rk4_temp, double curtime, double step ){

    C_DCOPY(rk4_dim,rk4_soln,1,rk4_temp,1);
    C_DAXPY(rk4_dim,step * time_step , rk4_in, 1, rk4_temp, 1);

    memset((void*)rk4_out,'\0',rk4_dim*sizeof(double));

    e_dip_x = 0.0;
    e_dip_y = 0.0;
    e_dip_z = 0.0;
    for (int i = 0; i < nmo; i++) {
        for (int j = 0; j < nmo; j++) {
            e_dip_x += rk4_temp[offset_dre+i*nmo+j] * dipole[0]->pointer()[i+nfzc][j+nfzc];
            e_dip_y += rk4_temp[offset_dre+i*nmo+j] * dipole[1]->pointer()[i+nfzc][j+nfzc];
            e_dip_z += rk4_temp[offset_dre+i*nmo+j] * dipole[2]->pointer()[i+nfzc][j+nfzc];
        }
    }
    e_dip_x *= 2.0;
    e_dip_y *= 2.0;
    e_dip_z *= 2.0;

    dipole_p_x = 0.0;
    dipole_p_y = 0.0;
    dipole_p_z = 0.0;
         
    temp_x->zero();
    temp_y->zero();
    temp_z->zero();

    for (int A=0; A<nS; A++){
        for (int B=0; B<nS; B++){
            for (int C=0; C<nS; C++){

                /* JJF Note:  only 1 component of density matrix for the NP */
                temp_x->pointer()[A][B] += rk4_temp[offset_dre_plasmon+A*nS+C]*Dip_x->pointer()[C][B];
                temp_y->pointer()[A][B] += rk4_temp[offset_dre_plasmon+A*nS+C]*Dip_y->pointer()[C][B];
                temp_z->pointer()[A][B] += rk4_temp[offset_dre_plasmon+A*nS+C]*Dip_z->pointer()[C][B];

            }
        }
    }

    for (int A=0; A<nS; A++){
        dipole_p_x += temp_x->pointer()[A][A];
        dipole_p_y += temp_y->pointer()[A][A];
        dipole_p_z += temp_z->pointer()[A][A];
    } 

    // kout = f( t( n + mh ) , y( n ) + mh kin)

    //ExtField(t); 
    ExtField(curtime);

    // electronic part
    ElectronicContribution(&rk4_temp[offset_dre],&rk4_temp[offset_dim],&rk4_out[offset_dre],&rk4_out[offset_dim]);

    // 3 components of plasmon part - JJFNOte only 1 component of density matrix for NP
    PlasmonContribution(&rk4_temp[offset_dre_plasmon],&rk4_temp[offset_dim_plasmon],&rk4_out[offset_dre_plasmon],&rk4_out[offset_dim_plasmon],Dip_x,Hp_x,polarization[0]);
    PlasmonContribution(&rk4_temp[offset_dre_plasmon],&rk4_temp[offset_dim_plasmon],&rk4_out[offset_dre_plasmon],&rk4_out[offset_dim_plasmon],Dip_y,Hp_y,polarization[1]);
    PlasmonContribution(&rk4_temp[offset_dre_plasmon],&rk4_temp[offset_dim_plasmon],&rk4_out[offset_dre_plasmon],&rk4_out[offset_dim_plasmon],Dip_z,Hp_z,polarization[2]);

    // 3 components of interaction term - JJFNote - only 1 component of NP density matrix
    InteractionContribution(&rk4_temp[offset_dre],          &rk4_temp[offset_dim],          &rk4_out[offset_dre],          &rk4_out[offset_dim],
                            &rk4_temp[offset_dre_plasmon],&rk4_temp[offset_dim_plasmon],&rk4_out[offset_dre_plasmon],&rk4_out[offset_dim_plasmon],
                            Hp_int_x,
                            dipole_pot_x,
                            e_dip_x,
                            dipole_p_x);

    InteractionContribution(&rk4_temp[offset_dre],          &rk4_temp[offset_dim],          &rk4_out[offset_dre],          &rk4_out[offset_dim],
                            &rk4_temp[offset_dre_plasmon],&rk4_temp[offset_dim_plasmon],&rk4_out[offset_dre_plasmon],&rk4_out[offset_dim_plasmon],
                            Hp_int_y,
                            dipole_pot_y,
                            e_dip_y,
                            dipole_p_y);

    InteractionContribution(&rk4_temp[offset_dre],          &rk4_temp[offset_dim],          &rk4_out[offset_dre],          &rk4_out[offset_dim],
                            &rk4_temp[offset_dre_plasmon],&rk4_temp[offset_dim_plasmon],&rk4_out[offset_dre_plasmon],&rk4_out[offset_dim_plasmon],
                            Hp_int_z,
                            dipole_pot_z,
                            e_dip_z,
                            dipole_p_z);
    // JJFNote - only 1 component of NP density matrix
    BuildLindblad(&rk4_temp[offset_dre_plasmon],&rk4_temp[offset_dim_plasmon],&rk4_out[offset_dre_plasmon],&rk4_out[offset_dim_plasmon]);
    BuildLindblad(&rk4_temp[offset_dre_plasmon],&rk4_temp[offset_dim_plasmon],&rk4_out[offset_dre_plasmon],&rk4_out[offset_dim_plasmon]);
    BuildLindblad(&rk4_temp[offset_dre_plasmon],&rk4_temp[offset_dim_plasmon],&rk4_out[offset_dre_plasmon],&rk4_out[offset_dim_plasmon]);

}

void TDHF::FFTW(){

    fftw_plan p;
    for (long int i=0; i<extra_pts; i++){
        corr_func[fftw_iter+i][0] = 0.0;
        corr_func[fftw_iter+i][1] = 0.0;
    }
    p = fftw_plan_dft_1d((int)(extra_pts+fftw_iter),corr_func,corr_func,FFTW_BACKWARD,FFTW_ESTIMATE);
    fftw_execute(p);
    fftw_destroy_plan(p);
    
    fftw_plan p2;
    for (long int i=0; i<extra_pts; i++){
        corr_func2[fftw_iter+i][0] = 0.0;
        corr_func2[fftw_iter+i][1] = 0.0;
    }
    p2 = fftw_plan_dft_1d((int)(extra_pts+fftw_iter),corr_func2,corr_func2,FFTW_BACKWARD,FFTW_ESTIMATE);
    fftw_execute(p2);
    fftw_destroy_plan(p2);
    
    fftw_plan p3;
    for (long int i=0; i<extra_pts; i++){
        corr_func3[fftw_iter+i][0] = 0.0;
        corr_func3[fftw_iter+i][1] = 0.0;
    }
    p3 = fftw_plan_dft_1d((int)(extra_pts+fftw_iter),corr_func3,corr_func3,FFTW_BACKWARD,FFTW_ESTIMATE);
    fftw_execute(p3);
    fftw_destroy_plan(p3);
    
    fftw_plan p4;
    for (long int i=0; i<extra_pts; i++){
        corr_func4[fftw_iter+i][0] = 0.0;
        corr_func4[fftw_iter+i][1] = 0.0;
    }
    p4 = fftw_plan_dft_1d((int)(extra_pts+fftw_iter),corr_func4,corr_func4,FFTW_BACKWARD,FFTW_ESTIMATE);
    fftw_execute(p4);
    fftw_destroy_plan(p4);
    
    fftw_plan p5;
    for (long int i=0; i<extra_pts; i++){
        corr_func5[fftw_iter+i][0] = 0.0;
        corr_func5[fftw_iter+i][1] = 0.0;
    }
    p5 = fftw_plan_dft_1d((int)(extra_pts+fftw_iter),corr_func5,corr_func5,FFTW_BACKWARD,FFTW_ESTIMATE);
    fftw_execute(p5);
    fftw_destroy_plan(p5);
    
    fftw_plan p6;
    for (long int i=0; i<extra_pts; i++){
        corr_func6[fftw_iter+i][0] = 0.0;
        corr_func6[fftw_iter+i][1] = 0.0;
    }
    p6 = fftw_plan_dft_1d((int)(extra_pts+fftw_iter),corr_func6,corr_func6,FFTW_BACKWARD,FFTW_ESTIMATE);
    fftw_execute(p6);
    fftw_destroy_plan(p6);
    
    fftw_plan p7;
    for (long int i=0; i<extra_pts; i++){
        corr_func7[fftw_iter+i][0] = 0.0;
        corr_func7[fftw_iter+i][1] = 0.0;
    }
    p7 = fftw_plan_dft_1d((int)(extra_pts+fftw_iter),corr_func7,corr_func7,FFTW_BACKWARD,FFTW_ESTIMATE);
    fftw_execute(p7);
    fftw_destroy_plan(p7);
    
    fftw_plan p8;
    for (long int i=0; i<extra_pts; i++){
        corr_func8[fftw_iter+i][0] = 0.0;
        corr_func8[fftw_iter+i][1] = 0.0;
    }
    p8 = fftw_plan_dft_1d((int)(extra_pts+fftw_iter),corr_func8,corr_func8,FFTW_BACKWARD,FFTW_ESTIMATE);
    fftw_execute(p8);
    fftw_destroy_plan(p8);
    
    fftw_plan p9;
    for (long int i=0; i<extra_pts; i++){
        corr_func9[fftw_iter+i][0] = 0.0;
        corr_func9[fftw_iter+i][1] = 0.0;
    }
    p9 = fftw_plan_dft_1d((int)(extra_pts+fftw_iter),corr_func9,corr_func9,FFTW_BACKWARD,FFTW_ESTIMATE);
    fftw_execute(p9);
    fftw_destroy_plan(p9);

}

// output absorption spectrum
void TDHF::Spectrum(){
  int i;
  double val,valr,vali,twopi = 2.0*M_PI;
  double w;
  outfile->Printf("\n");
  outfile->Printf("        ***********************************************************\n");
  outfile->Printf("        *                                                         *\n");
  outfile->Printf("        *                    Emission spectrum                    *\n");
  outfile->Printf("        *         as computed by the Fourier transform of         *\n");
  outfile->Printf("        *                   dipole acceleration                   *\n");
  outfile->Printf("        *                                                         *\n");
  outfile->Printf("        *     I(w) = |FourierTransform ( d^2 D(t) / dt^2 )|^2     *\n");
  outfile->Printf("        *                                                         *\n");
  outfile->Printf("        ***********************************************************\n");
  outfile->Printf("\n");
  outfile->Printf("                                w(eV)");
  outfile->Printf("                 I(w)\n");


  // fourier transform:
  int nfreq = 5001;
  double maxfreq = 30*0.08188379587298;
  double df = maxfreq / (nfreq - 1);
  double eps_0 = 1.0 / (4.0 * M_PI);
  eps_med = options_.get_double("EPSILON_M");
  //double eps_med = eps_0; //2.25;
  for (int i=1; i<(int)(fftw_iter+extra_pts); i++){
      w    = twopi*i/((extra_pts+fftw_iter)*time_step);
      if (w*pc_hartree2ev>max_freq) break;
      valr = corr_func[i][0]/fftw_iter;
      vali = corr_func[i][1]/fftw_iter;
      val  = sqrt(valr*valr + vali*vali);


      double e_dip_x_r   = corr_func[i][0]/fftw_iter;
      double e_dip_x_i   = corr_func[i][1]/fftw_iter;
      double e_dip_y_r   = corr_func2[i][0]/fftw_iter;
      double e_dip_y_i   = corr_func2[i][1]/fftw_iter;
      double e_dip_z_r   = corr_func3[i][0]/fftw_iter;
      double e_dip_z_i   = corr_func3[i][1]/fftw_iter;

      double p_dip_x_r   = corr_func4[i][0]/fftw_iter;
      double p_dip_x_i   = corr_func4[i][1]/fftw_iter;
      double p_dip_y_r   = corr_func5[i][0]/fftw_iter;
      double p_dip_y_i   = corr_func5[i][1]/fftw_iter;
      double p_dip_z_r   = corr_func6[i][0]/fftw_iter;
      double p_dip_z_i   = corr_func6[i][1]/fftw_iter;

      double dx_r = e_dip_x_r + p_dip_x_r;
      double dx_i = e_dip_x_i + p_dip_x_i;

      double dy_r = e_dip_y_r + p_dip_y_r;
      double dy_i = e_dip_y_i + p_dip_y_i;

      double dz_r = e_dip_z_r + p_dip_z_r;
      double dz_i = e_dip_z_i + p_dip_z_i;

      double ave_d_r = (dx_r + dy_r + dz_r)/3.0;
      double ave_d_i = (dx_i + dy_i + dz_i)/3.0;

      double field_x_r   = corr_func7[i][0]/fftw_iter;
      double field_x_i   = corr_func7[i][1]/fftw_iter;
      double field_y_r   = corr_func8[i][0]/fftw_iter;
      double field_y_i   = corr_func8[i][1]/fftw_iter;
      double field_z_r   = corr_func9[i][0]/fftw_iter;
      double field_z_i   = corr_func9[i][1]/fftw_iter;

      double field_x_mag = sqrt(field_x_r*field_x_r + field_x_i*field_x_i);
      double field_y_mag = sqrt(field_y_r*field_y_r + field_y_i*field_y_i);
      double field_z_mag = sqrt(field_z_r*field_z_r + field_z_i*field_z_i);

      double ave_f_r = (field_x_r + field_y_r + field_z_r)/3.0;
      double ave_f_i = (field_x_i + field_y_i + field_z_i)/3.0;

      double total_field_sq = field_x_r*field_x_r
                            + field_y_r*field_y_r
                            + field_z_r*field_z_r
                            + field_x_i*field_x_i
                            + field_y_i*field_y_i
                            + field_z_i*field_z_i;

      double total_field_sq_x = field_x_r*field_x_r + field_x_i*field_x_i;
      double total_field_sq_y = field_y_r*field_y_r + field_y_i*field_y_i;
      double total_field_sq_z = field_z_r*field_z_r + field_z_i*field_z_i;


//printf("%20.12le\n",total_field_sq);

      // From Gray's paper: Phys. Rev. B 88, 075411 (2013)


      double mol_sca_x = pow((w / 137.03),4) * (e_dip_x_i * e_dip_x_i + e_dip_x_r * e_dip_x_r) /(6.0*M_PI*eps_0*eps_0*(total_field_sq));
      double mol_sca_y = pow((w / 137.03),4) * (e_dip_y_i * e_dip_y_i + e_dip_y_r * e_dip_y_r) /(6.0*M_PI*eps_0*eps_0*(total_field_sq));
      double mol_sca_z = pow((w / 137.03),4) * (e_dip_z_i * e_dip_z_i + e_dip_z_r * e_dip_z_r) /(6.0*M_PI*eps_0*eps_0*(total_field_sq));

      double mol_abs_x = w*(e_dip_x_i*field_x_r - e_dip_x_r*field_x_i)/(total_field_sq*sqrt(eps_med)*eps_0*137.03);
      double mol_abs_y = w*(e_dip_y_i*field_y_r - e_dip_y_r*field_y_i)/(total_field_sq*sqrt(eps_med)*eps_0*137.03);
      double mol_abs_z = w*(e_dip_z_i*field_z_r - e_dip_z_r*field_z_i)/(total_field_sq*sqrt(eps_med)*eps_0*137.03);

      double mol_ext_x = mol_sca_x + mol_abs_x;
      double mol_ext_y = mol_sca_y + mol_abs_y;
      double mol_ext_z = mol_sca_z + mol_abs_z;

      double mol_sca = (mol_sca_x + mol_sca_y + mol_sca_z);
      double mol_abs = (mol_abs_x + mol_abs_y + mol_abs_z);
      double mol_ext = (mol_ext_x + mol_ext_y + mol_ext_z);

      double sca_x = pow((w / 137.03),4) * (dx_i * dx_i + dx_r * dx_r) /(6.0*M_PI*eps_0*eps_0*(total_field_sq));
      double sca_y = pow((w / 137.03),4) * (dy_i * dy_i + dy_r * dy_r) /(6.0*M_PI*eps_0*eps_0*(total_field_sq));
      double sca_z = pow((w / 137.03),4) * (dz_i * dz_i + dz_r * dz_r) /(6.0*M_PI*eps_0*eps_0*(total_field_sq));

      double abs_x = w*(dx_i*field_x_r - dx_r*field_x_i)/(total_field_sq*sqrt(eps_med)*eps_0*137.03);
      double abs_y = w*(dy_i*field_y_r - dy_r*field_y_i)/(total_field_sq*sqrt(eps_med)*eps_0*137.03);
      double abs_z = w*(dz_i*field_z_r - dz_r*field_z_i)/(total_field_sq*sqrt(eps_med)*eps_0*137.03);

      double ext_x = sca_x + abs_x;
      double ext_y = sca_y + abs_y;
      double ext_z = sca_z + abs_z;

      double sca = (sca_x + sca_y + sca_z);
      double abs = (abs_x + abs_y + abs_z);
      double ext = (ext_x + ext_y + ext_z);

      //double sca = pow((w / 137.03),4) * (ave_d_i * ave_d_i + ave_d_r * ave_d_r) / (6.0*M_PI*eps_0*(total_field_sq));
      //double abs = w*(ave_d_i*ave_f_r - ave_d_r*ave_f_i)/(total_field_sq*sqrt(eps_med)*eps_0*137.03);
      //double ext = sca + abs; 



      // Taken from http://arxiv.org/abs/1312.0899v1

      //double sca = pow(w,4)*(dz_i * dz_i + dz_r * dz_r)/(6.0*M_PI*sqrt(eps_0)*137.03*(total_field_sq_z));

      //double ext = -w*(-dz_i*field_z_r + dz_r*field_z_i)/(total_field_sq_z*sqrt(eps_0));

      //double abs = ext - sca;

      //outfile->Printf("      @Frequency %20.12lf %20.12le %20.12le %20.12le \n",w*pc_hartree2ev,sca,abs,ext);

      //outfile->Printf("      @Frequency %20.12lf %20.12lf %20.12le %20.12le %20.12le %20.12le %20.12le %20.12le %20.12le %20.12le \n",w*pc_hartree2ev,sca_cross,mur,mui,mumr,mumi,mupr,mupi,er,ei);
      outfile->Printf("      @Frequency %20.12lf %20.12le %20.12le %20.12le %20.12le %20.12le %20.12le %20.12le %20.12le %20.12le %20.12le %20.12le %20.12le %20.12le %20.12le %20.12le \n",w*pc_hartree2ev,sca,abs,ext,sca_x,abs_x,ext_x,sca_y,abs_y,ext_y,sca_z,abs_z,ext_z,field_x_mag,field_y_mag,field_z_mag);
      outfile->Printf("      @MolecularComponent %20.12lf %20.12le %20.12le %20.12le %20.12le %20.12le %20.12le %20.12le %20.12le %20.12le %20.12le %20.12le %20.12le %20.12le %20.12le %20.12le \n",w*pc_hartree2ev,mol_sca,mol_abs,mol_ext,mol_sca_x,mol_abs_x,mol_ext_x,mol_sca_y,mol_abs_y,mol_ext_y,mol_sca_z,mol_abs_z,mol_ext_z,field_x_mag,field_y_mag,field_z_mag);
  }
}

void TDHF::TransformIntegralsFull() {

    long int o = ndoccact;
    long int v = nvirt;
    long int full = o+v+nfzc+nfzv;

    nQ = (int)Process::environment.globals["NAUX (CC)"];

    long int memory = Process::environment.get_memory();
    // subtract out 20 full*full + 250 MB to be sure we have enough room
    memory -= sizeof(double)* 20L * full * full - 250L * 1024L * 1024L;

    if ( memory < sizeof(double) * (2L*full*full*nQ) ) {
        throw PsiException("TDHF::TransformIntegrals: not enough memory",__FILE__,__LINE__);
    }

    double * myQmo = (double*)malloc(full*full*nQ*sizeof(double));
    memset((void*)myQmo,'\0',full*full*nQ*sizeof(double));

    double ** Ca = Ca_->pointer();

    // available memory:
    memory -= sizeof(double) * (2L*full*full*nQ);
    int ndoubles = memory / sizeof(double) / 2;
    if ( nso * nso * nQ < ndoubles ) ndoubles = nso*nso*nQ;

    double * buf1 = (double*)malloc(ndoubles * sizeof(double));
    double * buf2 = (double*)malloc(ndoubles * sizeof(double));
    memset((void*)buf1,'\0',ndoubles*sizeof(double));
    memset((void*)buf2,'\0',ndoubles*sizeof(double));

    // (Q|rs)
    std::shared_ptr<PSIO> psio(new PSIO());
    psio->open(PSIF_DCC_QSO,PSIO_OPEN_OLD);
    psio_address addr1  = PSIO_ZERO;
    psio_address addrvo = PSIO_ZERO;
    long int nrows = 1;
    long int rowsize = nQ;
    while ( rowsize*nso*nso > ndoubles ) {
        nrows++;
        rowsize = nQ / nrows;
        if (nrows * rowsize < nQ) rowsize++;
        if (rowsize == 1) break;
    }
    long int lastrowsize = nQ - (nrows - 1L) * rowsize;
    long int * rowdims = (long int *)malloc(nrows*sizeof(long int));
    for (int i = 0; i < nrows-1; i++) rowdims[i] = rowsize;
    rowdims[nrows-1] = lastrowsize;
    for (int row = 0; row < nrows; row++) {
        psio->read(PSIF_DCC_QSO,"Qso CC",(char*)&buf1[0],rowdims[row]*nso*nso*sizeof(double),addr1,&addr1);
        F_DGEMM('n','n',full,nso*rowdims[row],nso,1.0,&Ca[0][0],full,buf1,nso,0.0,buf2,full);
        for (int q = 0; q < rowdims[row]; q++) {
            for (int mu = 0; mu < nso; mu++) {
                C_DCOPY(full,buf2+q*nso*full+mu*full,1,buf1+q*nso*full+mu,nso);
            }
        }
        F_DGEMM('n','n',full,full*rowdims[row],nso,1.0,&Ca[0][0],full,buf1,nso,0.0,buf2,full);

        // Qmo
        #pragma omp parallel for schedule (static)
        for (int q = 0; q < rowdims[row]; q++) {
            for (int a = 0; a < full; a++) {
                for (int b = 0; b < full; b++) {
                    myQmo[(q+rowdims[0]*row)*full*full+a*full+b] = buf2[q*full*full+a*full+b];
                }
            }
        }
    }
    //add frozen-core contribution to oeis
    for (int i = nfzc; i < full; i++) {
        for (int j = nfzc; j < full; j++) {
            double dum = 0.0;
            for (int k = 0; k < nfzc; k++) {
                for (int q = 0; q < nQ; q++) {
                    dum += 2.0 * myQmo[q*full*full+i*full+j] * myQmo[q*full*full+k*full+k];
                    dum -=       myQmo[q*full*full+i*full+k] * myQmo[q*full*full+k*full+j];
                }
            }
            T->pointer()[i][j] += dum;
        }
    }

    free(myQmo);

    free(buf1);
    free(buf2);
    free(rowdims);
    psio->close(PSIF_DCC_QSO,1);
}
void TDHF::TransformIntegrals() {

    long int o = ndoccact;
    long int v = nvirt;
    long int full = o+v+nfzc+nfzv;

    nQ = (int)Process::environment.globals["NAUX (CC)"];

    long int memory = Process::environment.get_memory();
    // subtract out 20 nmo*nmo + 250 MB to be sure we have enough room
    memory -= sizeof(double)* 20L * nmo * nmo - 250L * 1024L * 1024L;

    if ( memory < sizeof(double) * (2L*nmo*nmo*nQ) ) {
        throw PsiException("TDHF::TransformIntegrals: not enough memory",__FILE__,__LINE__);
    }

    Qmo = (double*)malloc((nmo+nfzc)*(nmo+nfzc)*nQ*sizeof(double));
    memset((void*)Qmo,'\0',(nmo+nfzc)*(nmo+nfzc)*nQ*sizeof(double));

    Ire = (double*)malloc(nmo*nmo*nQ*sizeof(double));
    Iim = (double*)malloc( ( nmo * nmo > nQ ? nmo * nmo : nQ ) *sizeof(double));
    memset((void*)Ire,'\0',nmo*nmo*nQ*sizeof(double));
    memset((void*)Iim,'\0',(nmo*nmo>nQ ? nmo*nmo : nQ)*sizeof(double));

    double ** Ca = Ca_->pointer();

    // available memory:
    memory -= sizeof(double) * (2L*nmo*nmo*nQ);
    int ndoubles = memory / sizeof(double) / 2;
    if ( nso * nso * nQ < ndoubles ) ndoubles = nso*nso*nQ;

    double * buf1 = (double*)malloc(ndoubles * sizeof(double));
    double * buf2 = (double*)malloc(ndoubles * sizeof(double));
    memset((void*)buf1,'\0',ndoubles*sizeof(double));
    memset((void*)buf2,'\0',ndoubles*sizeof(double));

    // (Q|rs)
    std::shared_ptr<PSIO> psio(new PSIO());
    psio->open(PSIF_DCC_QSO,PSIO_OPEN_OLD);
    psio_address addr1  = PSIO_ZERO;
    psio_address addrvo = PSIO_ZERO;
    long int nrows = 1;
    long int rowsize = nQ;
    while ( rowsize*nso*nso > ndoubles ) {
        nrows++;
        rowsize = nQ / nrows;
        if (nrows * rowsize < nQ) rowsize++;
        if (rowsize == 1) break;
    }
    long int lastrowsize = nQ - (nrows - 1L) * rowsize;
    long int * rowdims = (long int*)malloc(nrows*sizeof(long int));
    for (int i = 0; i < nrows-1; i++) rowdims[i] = rowsize;
    rowdims[nrows-1] = lastrowsize;
    for (int row = 0; row < nrows; row++) {
        psio->read(PSIF_DCC_QSO,"Qso CC",(char*)&buf1[0],rowdims[row]*nso*nso*sizeof(double),addr1,&addr1);
        F_DGEMM('n','n',full,nso*rowdims[row],nso,1.0,&Ca[0][0],full,buf1,nso,0.0,buf2,full);
        for (int q = 0; q < rowdims[row]; q++) {
            for (int mu = 0; mu < nso; mu++) {
                C_DCOPY(full,buf2+q*nso*full+mu*full,1,buf1+q*nso*full+mu,nso);
            }
        }
        F_DGEMM('n','n',full,full*rowdims[row],nso,1.0,&Ca[0][0],full,buf1,nso,0.0,buf2,full);

        // Qmo
        #pragma omp parallel for schedule (static)
        for (int q = 0; q < rowdims[row]; q++) {
            for (int a = 0; a < nmo; a++) {
                for (int b = 0; b < nmo; b++) {
                    Qmo[(q+rowdims[0]*row)*nmo*nmo+a*nmo+b] = buf2[q*full*full+a*full+b];
                }
            }
        }
    }
    free(buf1);
    free(buf2);
    free(rowdims);
    psio->close(PSIF_DCC_QSO,1);
}

void TDHF::ElectronicContribution(double* tempr,double* tempi,double* kre,double* kim) {

    Fre->zero();
    Fim->zero();

    // F1 = J1 - K1 + h1 - mu.E
    BuildFock(tempr,tempi,true);
    F1re->copy(Fre_temp);
    F1im->copy(Fim_temp);

    F_DGEMM('n','n',nmo,nmo,nmo,1.0,&(F1im->pointer()[0][0]),nmo,tempr,nmo,0.0,kre,nmo);
    F_DGEMM('n','n',nmo,nmo,nmo,1.0,&(F1re->pointer()[0][0]),nmo,tempi,nmo,1.0,kre,nmo);
    F_DGEMM('n','n',nmo,nmo,nmo,-1.0,tempr,nmo,&(F1im->pointer()[0][0]),nmo,1.0,kre,nmo);
    F_DGEMM('n','n',nmo,nmo,nmo,-1.0,tempi,nmo,&(F1re->pointer()[0][0]),nmo,1.0,kre,nmo);

    F_DGEMM('n','n',nmo,nmo,nmo,-1.0,&(F1re->pointer()[0][0]),nmo,tempr,nmo,0.0,kim,nmo);
    F_DGEMM('n','n',nmo,nmo,nmo,1.0,&(F1im->pointer()[0][0]),nmo,tempi,nmo,1.0,kim,nmo);
    F_DGEMM('n','n',nmo,nmo,nmo,1.0,tempr,nmo,&(F1re->pointer()[0][0]),nmo,1.0,kim,nmo);
    F_DGEMM('n','n',nmo,nmo,nmo,-1.0,tempi,nmo,&(F1im->pointer()[0][0]),nmo,1.0,kim,nmo);

}

/*  This size of the one-particle basis for the NP is determined by nmax...
 *      the size is nmax^3.  This is for the particle-in-a-cube model
 *          for the one-electron states  */
void TDHF::OrderCubicBasis() {
  /*  particle in a cube quantum numbers  */
  int nx, ny, nz;

  int idx, l;
  /* variables to use for ordering orbitals in increasing energy  */
  int cond, Ecur, swap, c, d;

  for (nx=0; nx<nmax; nx++) {
    for (ny=0; ny<nmax; ny++) {
      for (nz=0; nz<nmax; nz++) {

        idx = nx*nmax*nmax + ny*nmax + nz;
        /* integer that energy of orbital psi_nx,ny,nz is proportional to */
        l = (nx+1)*(nx+1) + (ny+1)*(ny+1) + (nz+1)*(nz+1);
        NPOrbE[idx] = l;

      }
    }
  }

  /* Now sort orbitals in increasing order  */
  for (c=0; c< (nmax*nmax*nmax-1); c++) {
    for (d=0; d<nmax*nmax*nmax-c-1; d++) {

      if (NPOrbE[d] > NPOrbE[d+1]) {

        swap = NPOrbE[d];
        NPOrbE[d] = NPOrbE[d+1];
        NPOrbE[d+1] = swap;

      }
    }
  }
  /* Now match up orbital indices with energies so that
 *      orbitals are stored in increasing energy order  */
  c=0;
  do {

    Ecur = NPOrbE[c];
    nx = 0;
    do {

      nx++;
      ny=0;

      do {

        ny++;
        nz=0;

        do {

          nz++;
          cond = Ecur-(nx*nx + ny*ny + nz*nz);

          if (cond==0) {

            NPOrb_x[c] = nx;
            NPOrb_y[c] = ny;
            NPOrb_z[c] = nz;
            c++;
          }

        }while( Ecur==NPOrbE[c] && nz<nmax);

      }while( Ecur==NPOrbE[c] && ny<nmax);

    }while( Ecur==NPOrbE[c] && nx<nmax);

  }while(c<nmax*nmax*nmax);


}



/*  This size of the one-particle basis for the NP is determined by nmax...               This is for the particle-in-a-sphere model for the one-electron states
 *      we may choose to limit the orbital angular momentum to be l_max, which
 *          will reduce the number of one-particle states for a given nmax  */
int TDHF::OrderSphericalBasis() {

  int i, n, l, m, idx, Enl;
  int LENGTH, l_max;


  idx=0;
  /* Principle quantum number, n (1, 2, ...., n_max) */
  for (n=1; n<=nmax; n++) {
    if (n<4) {
      l_max=n;
    }
    else {
       l_max = 3;
    }
    /* angular momentum quantum number l (0, ..., n)
 *        at this stage we are limiting l to be <= 2 */
    for (l=0; l<l_max; l++) {
      for (m=-l; m<=l; m++) {

        Enl = (2*n+l+2)*(2*n+l+2);
        NPOrbE[idx]= Enl;
        NPOrb_n[idx] = n;
        NPOrb_l[idx] = l;
        NPOrb_m[idx] = m;
        idx++;

      }
    }
  }


  LENGTH = idx;
  return LENGTH;

}




double TDHF::NP_h(int p) {

  double fac;
  if (options_.get_str("NP_SHAPE") == "CUBIC") {

    fac = pi*pi/(2.*L*L);

  }
  /* energy prefactor for nanospheres is hbar^2 pi^2/(8 m R^2)  */
  else if (options_.get_str("NP_SHAPE") == "SPHERICAL") {

    fac = pi*pi/(8.*R*R);

  }
  /* if not specified, assume a nanocube  */
  else fac = pi*pi/(2.*L*L);

  return fac*NPOrbE[p];


}




/*  Returns value of normalized Legendre polynomial at value of theta
 *      given quantum number l and m  */
double TDHF::Legendre(int l, int m, double theta) {


  int mp;
  double ctheta, pfac, P;
  double y;

  mp = m;
  /* Legendre Polynomial function will only take positive values of m */
  if (m<0) {

     mp = abs(m);
  }
  /* Prefactor  */
  pfac = prefac( mp, l );

  /* cosine of theta  */
  ctheta = cos(theta);

  /* Legendre Polynomial P_l^m (cos(theta))  */
  P = plgndr( l, mp, ctheta);

  /* Spherical Harmonic = prefac*P_l^m(cos(theta))*exp(i*m*phi) */
  y = pfac*P;

  return y;
}



/*  Computes normalization constant for Spherical Harmonics for a given m and l */
double TDHF::prefac(int m, int l) {


  double p, num1, num2, denom1, denom2;

  num1 = 2*l+1;
  num2 = factorial( (l-m) );

  denom1 = 4*pi;
  denom2 = factorial( (l+m) );


  p = sqrt((num1/denom1)*(num2/denom2));

  return p;

}

/*  Computes factorials! */
double TDHF::factorial(int n) {
  int c;
  double result = 1;

  for (c = 1; c <= n; c++)
    result = result * c;

  return result;
}




/*  This is from Numerical Recipes in C!  */
double TDHF::plgndr(int l, int m, double x) {
/*  Computes the associated Legendre polynomial P m
 *      l (x). Here m and l are integers satisfying
 *          0 ≤ m ≤ l, while x lies in the range −1 ≤ x ≤ 1.
 *              void nrerror(char error_text[]);  */

  double fact,pll,pmm,pmmp1,somx2;

  int i,ll;

  if (m < 0 || m > l || fabs(x) > 1.0) {
    printf("Bad arguments in routine plgndr\n");
    exit(0);
  }

  pmm=1.0;   /*  Compute P^m_m .  */

  if (m > 0) {

    somx2=sqrt((1.0-x)*(1.0+x));
    fact=1.0;

    for (i=1;i<=m;i++) {

      pmm *= -fact*somx2;
      fact += 2.0;

    }
  }
  if (l == m)
    return pmm;

  else {    /* Compute P^m_m+1  */

    pmmp1=x*(2*m+1)*pmm;

    if (l == (m+1))
      return pmmp1;

    else {   /* Compute P^m_l, l>m+1 */

      for (ll=m+2;ll<=l;ll++) {

        pll=(x*(2*ll-1)*pmmp1-(ll+m-1)*pmm)/(ll-m);
        pmm=pmmp1;
        pmmp1=pll;

     }

     return pll;
   }
  }
}




/*  Uses asymptotic approximation to Spherical Bessel Function of
 *      quantum numnber n and l (see The Kraus and Schatz, JCP 79, 6130 (1983); doi: 10.1063/1.445794)
 *          Big R is radius of the particle, little r is current value of r variable
 *              Function returns value of the function at r  */
double TDHF::Bessel(double r, int n, int l) {

  double cterm1, cterm2, cterm3, pre, val;

  val = 0;

  if (r>R) val=0;

  else {

    pre = 2./( sqrt(R)*r);
    cterm1 = (2*n+l+2)*r/R;
    cterm2 = l+1;
    cterm3 = (pi/2)*(cterm1-cterm2);

    /*val = pre*cos(cterm3)/sqrt(2); */
    val = pre*cos(cterm3);
  }

  return val;
}



void TDHF::Spherical_Y(int l, int m, double theta, double phi, double *Yr, double *Yi) {

  int mp;
  double ctheta, pfac, P;
  double y;

  mp = m;
  /* Legendre Polynomial function will only take positive values of m */
  if (m<0) {

    mp = abs(m);
  }

  /* Prefactor for Y  */
  pfac = prefac( mp, l );

  /* cosine of theta  */
  ctheta = cos(theta);

  /* Legendre Polynomial P_l^m (cos(theta)) */
  P = plgndr( l, mp, ctheta);

  /* Spherical Harmonic = prefac*P_l^m(cos(theta))*exp(i*m*phi)
 *      complex y:  y = pfac*P*cexp(I*m*phi);  */
  *Yr = pfac*P*cos(m*phi);
  *Yi = pfac*P*sin(m*phi);

}


double TDHF::TDP_Z_Spherical(int p, int q) {

  int ni, nf, li, lf, mi, mf;
  int i, j, numpts;
  double r, theta, phi;
  double dr, dth;
  double bji, bjf, Yri, Yrf, Yii, Yif;
  double sum;
  /*double complex jsum;
 *     jsum = 0. + 0.*I; */

  sum = 0.;
  /* orbital quantum numbers associated with index p */
  ni = NPOrb_n[p-1];
  li = NPOrb_l[p-1];
  mi = NPOrb_m[p-1];

  /* orbital quantum numbers associated with index q  */
  nf = NPOrb_n[q-1];
  lf = NPOrb_l[q-1];
  mf = NPOrb_m[q-1];

  numpts = 1000;

  dr = R/numpts;
  dth = pi/numpts;

  if (mi==mf && ( (lf-li)==1 || (li-lf)==1) ) {

    for (i=0; i<=numpts; i++) {

      if (i==0) r=1e-9;

      else {
        r = dr*i;
      }
      bji = Bessel(r, ni, li);
      bjf = Bessel(r, nf, lf);

      for (j=0; j<=numpts; j++) {

        theta = j*dth;

          phi = 0.;

          Spherical_Y(li, mi, theta, phi, &Yri, &Yii);
          Spherical_Y(lf, mf, theta, phi, &Yrf, &Yif);

          /*sum += bji*bjf*r*r*(Yri + I*Yii)*(Yrf + I*Yif)*r*cos(theta)*sin(theta)*dt*dr;  */
          sum += bji*bjf*r*r*(Yri)*(Yrf)*r*cos(theta)*sin(theta)*dth*dr;
       }
     }

  /*  -e int Psi(ni, li, mi) r*cos(theta) * Psi(nf, lf, mf) dr dtheta dphi
 *    = -e * 2*pi * int j(ni, li) P_li (cos(theta)) r*cos(theta) * j(nf, lf) P_lf (cos(theta)) dr dtheta
 *
 *        Ignoring imaginary part!!
 *                return -2*pi*creal(jsum);  */
  return -2*pi*sum;
  /*mur = -2*pi*creal(sum);
 *    *mui = -2*pi*cimag(sum);  */

  }
  else {

    return 0.;

  }
}


double TDHF::TDP_Z_Cubic(int p, int q) {

  int nxi, nxf, nyi, nyf, nzi, nzf;
  double dipole_integral;
  double term1, term2;

  /* orbital quantum numbers associated with index p */
  nxi = NPOrb_x[p-1];
  nyi = NPOrb_y[p-1];
  nzi = NPOrb_z[p-1];

  /* orbital quantum numbers associated with index q */
  nxf = NPOrb_x[q-1];
  nyf = NPOrb_y[q-1];
  nzf = NPOrb_z[q-1];

  /* z-component: to be non-zero, it is required that
 *      nxi==nxf && nyi==nyf  */
  if (nxi==nxf && nyi==nyf) {

      term1 = pow(L,2)*(pi*(nzi-nzf)*sin(pi*(nzi-nzf))+cos(pi*(nzi-nzf))-1)/(pi*pi*pow((nzi-nzf),2));
      term2 = pow(L,2)*(pi*(nzi+nzf)*sin(pi*(nzi+nzf))+cos(pi*(nzi+nzf))-1)/(pi*pi*pow((nzi+nzf),2));
      dipole_integral = (1./L)*(term1-term2);

  }

  else {

    dipole_integral = 0.;

  }

  return dipole_integral;

}


double TDHF::TDP_Y_Cubic(int p, int q) {

  int nxi, nxf, nyi, nyf, nzi, nzf;
  double dipole_integral;
  double term1, term2;

  /* orbital quantum numbers associated with index p  */
  nxi = NPOrb_x[p-1];
  nyi = NPOrb_y[p-1];
  nzi = NPOrb_z[p-1];

  /* orbital quantum numbers associated with index q  */
  nxf = NPOrb_x[q-1];
  nyf = NPOrb_y[q-1];
  nzf = NPOrb_z[q-1];

  /* y-component: to be non-zero, it is required that
 *      nxi==nxf && nzi==nzf  */

  if (nxi==nxf && nzi==nzf) {

      term1 = pow(L,2)*(pi*(nyi-nyf)*sin(pi*(nyi-nyf))+cos(pi*(nyi-nyf))-1)/(pi*pi*pow((nyi-nyf),2));
      term2 = pow(L,2)*(pi*(nyi+nyf)*sin(pi*(nyi+nyf))+cos(pi*(nyi+nyf))-1)/(pi*pi*pow((nyi+nyf),2));
      dipole_integral = (1./L)*(term1-term2);

  }

  else {

    dipole_integral = 0.;

  }

  return dipole_integral;



}



double TDHF::TDP_X_Cubic(int p, int q) {

  int nxi, nxf, nyi, nyf, nzi, nzf;
  double dipole_integral;
  double term1, term2;

  /* orbital quantum numbers associated with index p */
  nxi = NPOrb_x[p-1];
  nyi = NPOrb_y[p-1];
  nzi = NPOrb_z[p-1];

  /* orbital quantum numbers associated with index q */
  nxf = NPOrb_x[q-1];
  nyf = NPOrb_y[q-1];
  nzf = NPOrb_z[q-1];

  /* x-component: to be non-zero, it is required that
 *   / nyi==nyf && nzi==nzf  */

  if (nyi==nyf && nzi==nzf) {

      term1 = pow(L,2)*(pi*(nxi-nxf)*sin(pi*(nxi-nxf))+cos(pi*(nxi-nxf))-1)/(pi*pi*pow((nxi-nxf),2));
      term2 = pow(L,2)*(pi*(nxi+nxf)*sin(pi*(nxi+nxf))+cos(pi*(nxi+nxf))-1)/(pi*pi*pow((nxi+nxf),2));
      dipole_integral = (1./L)*(term1-term2);

  }

  else {

    dipole_integral = 0.;

  }

  return dipole_integral;

}



double TDHF::TDPEval(int p, int q, char *component) {
  double TDM;
  TDM = 0.;
  if (options_.get_str("NP_SHAPE") == "CUBIC") {

    if (strcmp(component,"X")==0) {

      TDM = TDP_X_Cubic(p, q);

    }
    else if (strcmp(component,"Y")==0) {

      TDM = TDP_Y_Cubic(p, q);

    }
    else if (strcmp(component,"Z")==0) {

      TDM = TDP_Z_Cubic(p, q);

    }
    else {

      TDM = 0.;
      printf("  Warning:  Not a valid component option!");

    }
  }
  else if (options_.get_str("NP_SHAPE") == "SPHERICAL") {

    if (strcmp(component,"Z")==0) {

      TDM = TDP_Z_Spherical(p, q);

    }
    else {

      printf("  Warning:  Only z-component of TDM supported for spherical NPs\n");
      TDM = 0.;

    }

  }
  return TDM;
}





}}
