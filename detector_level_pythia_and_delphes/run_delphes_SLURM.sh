#!/bin/bash

#SBATCH -p lipq
#SBATCH --mem=2G

module load gcc63/madgraph/3.3.1

DELPHES_PATH=/cvmfs/sw.el7/gcc63/madgraph/3.3.1/b01/Delphes
source ${DELPHES_PATH}/DelphesEnv.sh

echo "SAMPLE_DIR: "${SAMPLE_DIR}
ls ${SAMPLE_DIR}
gzip -dv ${SAMPLE_DIR}/tag_1_pythia8_events.hepmc.gz
${DELPHES_PATH}/DelphesHepMC ${DELPHES_PATH}/cards/delphes_card_HLLHC.tcl ${SAMPLE_DIR}/delphes_events.root ${SAMPLE_DIR}/tag_1_pythia8_events.hepmc