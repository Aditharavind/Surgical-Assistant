# AI Doctor Assistant for Chest,lung,Brain Disease Detection

An AI-powered Doctor Assistant that aids in diagnosing chest conditions (Normal vs. Pneumonia) from X-ray images and provides actionable clinical recommendations. In future implementations, the system will be enhanced with a Large Language Model (LLM) integration to deliver more interactive and detailed diagnostic support.

## Overview

This project uses a Convolutional Neural Network (CNN) built with PyTorch to classify chest X-rays into **Normal** and **Pneumonia**. A user-friendly Streamlit interface is provided for testing the model and receiving step-by-step medical guidance. The dataset used in this project must be downloaded from Kaggle.

**Future Work:**  
- **LLM Integration:** Future versions will integrate a Large Language Model (LLM) to generate comprehensive clinical recommendations and natural language explanations to further support doctors in their decision-making.
- **Additional Diagnostic Modules:** Plans to expand the system to include brain tumor detection and lung cancer analysis.

## Features

- **Chest X-ray Classification:** Automatically classifies X-rays into Normal or Pneumonia.
- **Step-by-Step Clinical Guidance:** Provides a detailed step-by-step guide for the next medical actions, including recommended equipment.
- **Streamlit Interface:** Easy-to-use web interface for doctors to upload images and view predictions.
- **Future LLM Integration:** Plans to incorporate an LLM to provide enhanced clinical explanations and interactive support.

## Dataset

The chest X-ray dataset can be downloaded from Kaggle:  
[Chest X-ray Dataset (Kaggle)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)

After downloading, organize the dataset with the following directory structure:

