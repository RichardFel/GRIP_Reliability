# GRIP_Human_Activity_Patterns

The GRIP (beweeGsensoren voor mensen met chRonIsche Pijn) project, aims to use movement sensors (accelerometers) in order to model the complex activity patterns of people with Chronic Pain (CP).
Several challenges arise in this project, for example lack of enough labeled data that allows for a Machine Learning (ML) algorithm to properly learn about the activity patterns of people with CP.

The Data Science Pool (DSP) team collaborates with researchers from the [Lifestyle and Health](https://www.internationalhu.com/research/lifestyle-and-health) Department at the University of Applied Sciences in this effort.

In this repo we keep a centralized Sandbox, where we experiment with open source datasets in order to gather further understanding about the technicalities of working with sensor data, in our case accelerometers.

Many parts of the code are simple jupyter notebooks where we perform the experiments; we use openly available datasets and code for our tests.

## Overview

The project is designed to take raw accelerometer data, preprocess it, classify it into physical activity categories, and then calculate various physical activity features. The workflow includes:

- **Data Preprocessing**: Raw data cleaning and synchronization.
- **Activity Classification**: Classification into different physical intesitities: sedentairy, light, medium and high intensity.
- **Feature Calculation**: Daily features based on the classification bouts.

## Usage

Create data folder and add subfolders

- Set the sample frequency and sample size in the config file.
- Add data to the raw_data folder -> data/raw_data/subject/days/sensorfile.csv

Create virtual env

- python -m venv venv
- pip install -r requirements.txt

run files

- Check settings in main.py
- Run main.py

## Features

- **Sensor Data Processing**: Process raw accelerometer data to extract movement patterns.
- **Activity Classification**: Classify physical activities into predefined categories using machine learning or rule-based models.
- **Daily Activity Features**: Calculate total time in each physical activity category and complexity of movement patterns during the day.
- **Customizable Pipelines**: Easy to modify or add additional feature extraction or classification methods.

## Data Requirements

- **Input**: The input should be raw accelerometer data in the following format:

  - **Sensor settings**: Settings in the first 8 lines
  - **Timestamp**:
  - **X, Y, Z acceleration**: Raw acceleration data in 3 axes.
  - **Timestamp True time**: Timestamp true time in miliseconds (before and after sensor data)

  Example:

  ```csv
  serial,HW Version,FW Version,SW Version
  2201001028,2,v1.17-278-g7566d107d on 2021-12-22,0xc870bf3
  settings,frequency,scale,decimation
  FIFO,12.5
  Accelerometer,12.5,4,1
  Gyroscope,0,500,1
  timestamp,high res,True,1
  Timestamp,Accelerometer,,,Gyroscope,RTC timestamp
  ,X,Y,Z,X,Y,Z,
  ,,,,,,,492985011
  1046,-0.2926,-0.2148,0.9441999,,,,
  1800,-0.2779,-0.2329,0.9501,,,,
  2554,-0.2594,-0.2248,0.9632,,,,
  3308,-0.2804,-0.2386,0.9899,,,,
  4062,-0.2712,-0.2522,0.9734,,,,
  ```

# Project Structure

This repository contains the following directories and files:

## Directories

- **config/**  
  Contains configuration files (e.g., `.json`, `.yaml`, or `.ini`) to manage project settings and parameters.

- **data/**  
  Directory for storing raw or processed data used in the project.

- **Figures/**  
  Used to store plots, charts, or visualizations generated during the analysis.

- **logging/**  
  Holds logging-related configurations or log files that track the execution of the project.

- **models/**  
  Stores machine learning models, including pre-trained models or model outputs.

- **Results/**  
  Contains outputs of the project, such as analysis results, processed data, or final reports.

- **src/**  
  Source code directory where the main project scripts are implemented.

- **venv/**  
  Virtual environment folder containing the Python packages and dependencies for the project, ensuring an isolated environment.

## Files

- **.gitignore**  
  Specifies files and directories that should be ignored by Git (e.g., environment files, temporary data).

- **10sec.png** & **test.png**  
  Image files, possibly used for example visualizations or testing purposes.

- **LICENSE**  
  The license file that defines the legal terms under which the project can be used or distributed.

- **main.py**  
  The main script for running the project. This file likely initiates the data processing pipeline, including sensor data preprocessing, activity classification, and feature extraction.

- **README.md**  
  Project documentation, providing an overview of the project, how to install and use it, and any necessary details about its structure.

- **reliability.py**  
  Script dedicated to running reliability checks or tests, ensuring robustness and consistency in data processing or model performance.

- **requirements.txt**  
  A list of Python packages and dependencies required by the project. Use this file to install dependencies with `pip`.
