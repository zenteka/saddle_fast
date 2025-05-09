#!/bin/bash
# Author Karol Piera
# Run saddle analysis from the terminal.
SCRIPT_DIR=$(dirname $0)
set -e

# Hardcoded, same as in Luca's LCS pipeline
HIC_RES=1000000
CALDER_RES=50000
N_TILES=16
EXCLUDED_CHROMOSOMES=''

usage() {
  echo "Usage: $0 -i <input> -c <compartments> -o <outpath>"
  echo "  -i <input>         Path to the input .mcool file"
  echo "  -c <compartments>  Path to the compartments file"
  echo "  -o <outpath>      Path to the output directory"
  echo "  -g <genome>       Genome version (hg38 or hg19). Genome is used to exclude chromosomes from the analysis"
  echo "  -h                Display this help message"
}

while getopts "i:c:o:g:" opt; do
  case $opt in
      i) COOLPATH="$OPTARG";; # MCOOL path e.g. inter_30.mcool
      c) COMPARTMENTS="$OPTARG" ;; # COMPARTMENTS
      g) genome="$OPTARG"
	 if [[ $genome == "hg38" ]]; then
	     EXCLUDED_CHROMOSOMES=$(cat ${SCRIPT_DIR}/ref/hg38_chroms_to_exclude_default.txt | tr '\n' ',')
	     echo "Excluding default chromosomes from hg38"
	 elif [[ $genome == "hg19" ]]; then
	     eXCLUDED_CHROMOSOMES=$(cat ${SCRIPT_DIR}/ref/hg19_chroms_to_exclude_default.txt | tr '\n' ',')
	     echo "Excluding default chromosomes from hg19"
	 else
	     echo "Genome not recognized. Please make sure that ref diretory contains fiiles:"
	     echo "hg38_chroms_to_exclude_default.txt"
	     echo "hg19_chroms_to_exclude_default.txt"
	     echo "Alternatively, you can specify the chromosomes to exclude in the script"
	     
	 fi
	 ;;
      o) OUTPATH="$OPTARG" ;;
      h) usage
	 exit 0 ;;
      \?) echo "Invalid option -$OPTARG" >&2
	  usage
	  exit 1 ;;
  esac
done


python ${SCRIPT_DIR}/src/saddle_fast.py ${COOLPATH} ${COMPARTMENTS} \
       ${OUTPATH} \
       --hic_resolution ${HIC_RES} \
       --compartments_resolution ${CALDER_RES} \
       --n_tiles ${N_TILES} \
       --excludeChroms ${EXCLUDED_CHROMOSOMES}



