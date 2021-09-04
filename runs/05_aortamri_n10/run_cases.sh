#!/bin/bash

cd CS
qsub run_recovery_CS_BE.sh
qsub run_recovery_CS_VG_n0_p25.sh
qsub run_recovery_CS_VG_n0_p50.sh
qsub run_recovery_CS_VG_n0_p75.sh
cd ..

cd CSDEB
qsub run_recovery_CSDEB_BE.sh
qsub run_recovery_CSDEB_VG_n0_p25.sh
qsub run_recovery_CSDEB_VG_n0_p50.sh
qsub run_recovery_CSDEB_VG_n0_p75.sh
cd ..

cd OMP
qsub run_recovery_OMP_BE.sh
qsub run_recovery_OMP_VG_n0_p25.sh
qsub run_recovery_OMP_VG_n0_p50.sh
qsub run_recovery_OMP_VG_n0_p75.sh
cd ..