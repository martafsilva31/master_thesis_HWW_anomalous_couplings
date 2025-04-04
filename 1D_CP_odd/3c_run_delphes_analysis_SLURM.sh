#!/bin/bash
#SBATCH -p lipq
#SBATCH -n 4
#SBATCH --mem=16G
# INPUT = delphes_analysis.py

module load python/3.7.2

echo ${MAIN_DIR}
echo ${SAMPLE_DIR}
echo ${BENCHMARK}
echo ${SETUP_FILE}
echo "DO_DELPHES: " ${DO_DELPHES} "DELPHES_CARD: " ${DELPHES_CARD}

if [ "$DO_DELPHES" = "True" ]; then
python3 3a_run_delphes_analysis.py --main_dir ${MAIN_DIR} --sample_dir ${SAMPLE_DIR} --setup_file ${SETUP_FILE} --benchmark ${BENCHMARK} --do_delphes --delphes_card ${DELPHES_CARD}
else
python3 3a_run_delphes_analysis.py --main_dir ${MAIN_DIR} --sample_dir ${SAMPLE_DIR} --setup_file ${SETUP_FILE} --benchmark ${BENCHMARK}
fi

