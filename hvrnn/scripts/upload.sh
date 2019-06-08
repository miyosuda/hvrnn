#!/bin/sh

USER_NAME="xxxxx"
IP="x.x.x.x"

#scp *.py ${USER_NAME}@${IP}:/home/${USER_NAME}/hvrnn/hvrnn
#scp ../data/oculomotor/*.py ${USER_NAME}@${IP}:/home/${USER_NAME}/hvrnn/data/oculomotor
#scp ../data/oculomotor/*.npz ${USER_NAME}@${IP}:/home/${USER_NAME}/hvrnn/data/oculomotor

echo ../data/oculomotor/*.npz ${USER_NAME}@${IP}:/home/${USER_NAME}/hvrnn/data/oculomotor
