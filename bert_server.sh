#!/bin/bash
#
#SBATCH --job-name=bert_server
#SBATCH --output=res_%j.txt  # output file
#SBATCH -e res_%j.err        # File to which STDERR will be written
#SBATCH --partition=titanx-short # Partition to submit to 
#
#SBATCH --ntasks=1
#SBATCH --time=03:59:59         # Runtime in D-HH:MM
#SBATCH --mem-per-cpu=1000    # Memory in MB per cpu allocated

bert-serving-start -model_dir /home/cmondragonch/uncased_L-12_H-768_A-12/ -num_worker=4 -max_seq_len=30 &
sleep 10
python embs_to_json.py
