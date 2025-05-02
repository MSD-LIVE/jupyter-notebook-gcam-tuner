FROM ghcr.io/msd-live/jupyter/python-notebook:latest as builder

COPY notebooks /home/jovyan/notebooks