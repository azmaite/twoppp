#!/bin/bash

# Check if the number of arguments passed is correct
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 input_file camera_file"
    exit 1
fi

# Assign arguments to variables
input_file=$1
camera_num=$2


# make the conda command known to the current shell by performing source /.../conda.sh
BASE=$(conda info | grep -i 'base environment' | cut -d':' -f 2 | cut -d'(' -f 1 | cut -c2- | rev | cut -c3- | rev)
BASE="${BASE}/etc/profile.d/conda.sh"
echo "will source conda base directory: ${BASE}"
source $BASE

OLD_ENV=$CONDA_DEFAULT_ENV
echo "old environment: ${OLD_ENV}"
echo "will activate 'sleap' environment"
conda activate sleap

# TODO: make this more flexible
#sleap_model="/home/jbraun/projects/twoppp/twoppp/behaviour/sleap_model"
sleap_model="/mnt/upramdya_files/AZCORRA_Maite/Other/sleap/new_model_LR/models/240719_180539.single_instance.n=802"

# Construct camera filename based on camera_num
camera="camera_${camera_num}.mp4"

while read d; do
  f="${d}/${camera}"
  # echo "$f"
  if test -f "$f"; then
    if test -f "${d}/sleap_output.h5"; then
      echo "already ran sleap on $f. Skipping"
    else
      echo "will run sleap on $f."
      sleap-track $f -m "$sleap_model" --verbosity rich --batch_size 32
      sleap-convert --format analysis -o "${d}/sleap_output.h5" "${f}.predictions.slp" 
    fi
  else
    echo "$f does not exist"
  fi

done < ${1}
# sleap_dirs.txt

conda activate $OLD_ENV
