#!/bin/bash
# script to remove files created to submit a calculation to the cluster

echo Scrubbing the directory!

#jobs
rm ./*.job

echo DONE!