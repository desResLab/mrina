#!/bin/bash

cd CS
qsub run_recovery_CS_VG_n5_p85.sh
qsub run_recovery_CS_VG_n5_p90.sh
qsub run_recovery_CS_VG_n5_p95.sh
qsub run_recovery_CS_VG_n10_p85.sh
qsub run_recovery_CS_VG_n10_p90.sh
qsub run_recovery_CS_VG_n10_p95.sh
qsub run_recovery_CS_VG_n30_p85.sh
qsub run_recovery_CS_VG_n30_p90.sh
qsub run_recovery_CS_VG_n30_p95.sh
cd ..

cd CSDEB
qsub run_recovery_CSDEB_VG_n5_p85.sh
qsub run_recovery_CSDEB_VG_n5_p90.sh
qsub run_recovery_CSDEB_VG_n5_p95.sh
qsub run_recovery_CSDEB_VG_n10_p85.sh
qsub run_recovery_CSDEB_VG_n10_p90.sh
qsub run_recovery_CSDEB_VG_n10_p95.sh
qsub run_recovery_CSDEB_VG_n30_p85.sh
qsub run_recovery_CSDEB_VG_n30_p90.sh
qsub run_recovery_CSDEB_VG_n30_p95.sh
cd ..

cd OMP
qsub run_recovery_OMP_VG_n5_p85.sh
qsub run_recovery_OMP_VG_n5_p90.sh
qsub run_recovery_OMP_VG_n5_p95.sh
qsub run_recovery_OMP_VG_n10_p85.sh
qsub run_recovery_OMP_VG_n10_p90.sh
qsub run_recovery_OMP_VG_n10_p95.sh
qsub run_recovery_OMP_VG_n30_p85.sh
qsub run_recovery_OMP_VG_n30_p90.sh
qsub run_recovery_OMP_VG_n30_p95.sh
cd ..