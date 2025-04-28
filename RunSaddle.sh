#!/bin/bash
# Author Karol Piera
# Run saddle analysis from the terminal.
SCRIPT_DIR=$(dirname $0)

set -e
# Hardcoded, same as in Luca's LCS pipeline
HIC_RES=1000000
CALDER_RES=50000
N_TILES=16
# Here we defien excluded. Modify if needed. Script parses comma separated list, make sure it passed as such.
EXCLUDED_CHROMOSOMES=$(cat ${SCRIPT_DIR}/ref/hg19_chroms_to_exclude_default.txt | tr '\n' ',')

usage() {
  echo "Usage: $0 -i <input> -c <compartments> -o <outpath>"
  echo "  -i <input>         Path to the input .mcool file"
  echo "  -c <compartments>  Path to the compartments file"
  echo "  -o <outpath>      Path to the output directory"
}

while getopts "i:c:o:" opt; do
  case $opt in
      i) COOLPATH="$OPTARG";; # MCOOL path e.g. inter_30.mcool
      c) COMPARTMENTS="$OPTARG" ;; # COMPARTMENTS 
      o) OUTPATH="$OPTARG" ;;
      h) echo "Usage: $0 -i <input> -c <compartments> -o <outpath>"
       exit 1 ;;
  esac
done


python src/saddle_fast.py ${COOLPATH} ${COMPARTMENTS} \
       ${OUTPATH} \
       --hic_resolution ${HIC_RES} \
       --compartments_resolution ${CALDER_RES} \
       --n_tiles ${N_TILES} \
       --excludeChroms ${EXCLUDED_CHROMOSOMES}



