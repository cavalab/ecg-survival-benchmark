#!/bin/bash

for file in ./Jobs/*.txt; do 
	printf "$file"
	sbatch "$file"
done


