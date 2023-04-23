#CAMB Makefile
#Edit for your compiler
#Note there are many ifc versions, some of which behave oddly

#Intel , -openmp toggles mutli-processor:
#note version 10.0 gives wrong result for lensed when compiled with -openmp [fixed in 10.1]

####works on Mac OSX v 10.5.6 with ifort 10.1.017 with uses 64-bit architecture by default.
CC = gcc
CFLAGS =
F90C     = ifort
FFLAGS = -qopenmp -O2 -ip -W0 -WB -fpp2 -vec-report0 -lgsl -lgslcblas -L/usr/local/lib
INCS   =
LIBS   = -lgsl -lgslcblas

##### Gfortran compiler Mac OSX using GSL from Fink
#CC = gcc
#CFLAGS =
#F90C     = gfortran
#FFLAGS = -O3 -ffixed-line-length-none -ffree-line-length-none -fopenmp
#INCS   = -I. -I/sw/include/
#LIBS = -L. -L/sw/lib/ -lgsl -lgslcblas


# Intel 9 on IA-64 (eg. COSMOS)
# (do "module load icomp90" before compiling)
#F90C = ifort
#FFLAGS = -openmp -fpp2 -w -O3 -ip -mP2OPT_hlo_prefetch=F

#Intel ifc, add -openmp for multi-processor (some have bugs):
#F90C     = ifc
#FFLAGS = -O2 -Vaxlib -ip -W0 -WB -quiet -fpp2
#some systems can can also add e.g. -tpp7 -xW

#G95 compiler
#F90C   = g95
#FFLAGS = -O2

#Gfortran compiler: if pre v4.3 add -D__GFORTRAN__
#F90C     = gfc
#FFLAGS =  -O2

#SGI, -mp toggles multi-processor. Use -O2 if -Ofast gives problems.
#F90C     = f90
#FFLAGS  = -Ofast -mp

#Digital/Compaq fortran, -omp toggles multi-processor
#F90C    = f90
#FFLAGS  = -omp -O4 -arch host -math_library fast -tune host -fpe1

#Absoft ProFortran, single processor:
#F90C     = f95
#FFLAGS = -O2 -cpu:athlon -s -lU77 -w -YEXT_NAMES="LCS" -YEXT_SFX="_"

#NAGF95, single processor:
#F90C     = f95
#FFLAGS = -DNAGF95 -O3

#PGF90
#F90C = pgf90
#FFLAGS = -O2 -DESCAPEBACKSLASH

#Sun V880
#F90C = mpf90
#FFLAGS =  -O4 -openmp -ftrap=%none -dalign -DMPI

#Sun parallel enterprise:
#F90C     = f95
#FFLAGS =  -O2 -xarch=native64 -openmp -ftrap=%none
#try removing -openmp if get bus errors. -03, -04 etc are dodgy.

#IBM XL Fortran, multi-processor (run gmake)
#F90C     = xlf90_r
#FFLAGS  = -DESCAPEBACKSLASH -DIBMXL -qsmp=omp -qsuffix=f=f90:cpp=F90 -O3 -qstrict -qarch=pwr3 -qtune=pwr3


#Files containing evolution equations initial power spectrum module
EQUATIONS     = equations
POWERSPECTRUM = power_tilt
REIONIZATION = reionization
RECOMBINATION = recfast

#Module doing non-linear scaling
NONLINEAR     = halofit

#Driver program
DRIVER        = inidriver.F90
#DRIVER        = sigma8.f90
#DRIVER        = tester.f90

#Settings for building camb_fits
#Location of FITSIO and name of library
FITSDIR       = /home/cpac/cpac-tools/lib
FITSLIB       = cfitsio
#Location of HEALPIX for building camb_fits
HEALPIXDIR    = /home/cpac/cpac-tools/healpix

CAMBLIB       = libcamb.a

#Shouldn't need to change anything else...

F90FLAGS      = $(FFLAGS)
HEALPIXLD     = -L$(HEALPIXDIR)/lib -lhealpix -L$(FITSDIR) -l$(FITSLIB)
FC            = $(F90C)

CAMBOBJ       = constants.o utils.o subroutines.o inifile.o $(POWERSPECTRUM).o $(RECOMBINATION).o $(REIONIZATION).o modules.o \
	bessels.o $(EQUATIONS).o $(NONLINEAR).o lensing.o cmbmain.o camb.o bsplinepk.o

default: camb

all: camb $(CAMBLIB)


subroutines.o: constants.o utils.o
$(POWERSPECTRUM): subroutines.o
$(RECOMBINATION).o: subroutines.o
$(REIONIZATION).o: constants.o
modules.o: $(REIONIZATION).o bsplinepk.o
bessels.o: modules.o
$(EQUATIONS): bessels.o
$(NONLINEAR).o:  modules.o
lensing.o: bessels.o
cmbmain.o: lensing.o
camb.o: cmbmain.o


camb: $(CAMBOBJ) $(DRIVER)
	$(F90C) $(F90FLAGS) $(CAMBOBJ) $(DRIVER) -o $@ $(LIBS)

$(CAMBLIB): $(CAMBOBJ)
	ar -r $@ $?

camb_fits: writefits.f90 $(CAMBOBJ) $(DRIVER)
	$(F90C) $(F90FLAGS) -I$(HEALPIXDIR)/include $(CAMBOBJ) writefits.f90 $(DRIVER) $(HEALPIXLD) -DWRITE_FITS -o $@

%.o: %.f90
	$(F90C) $(F90FLAGS) $(INCS) -c $*.f90

utils.o:
	$(F90C) $(F90FLAGS) $(INCS) -c utils.F90

bsplinepk.o:
	$(CC) $(CFLAGS) $(INCS) -c bsplinepk.c

clean:
	-rm -f *.o *.a *.d core *.mod
