# Neocognitron Implementation in Python

## Overview

This repository provides an implementation of the Neocognitron, a hierarchical neural network model designed for pattern recognition, originally proposed by Kunihiko Fukushima. This implementation includes visualization tools for inspecting the feature extraction process and supports network serialization using the pickle module.

## Features

✅ Implements a multi-layer Neocognitron network.

✅ Supports training with supervised and unsupervised approaches.

✅ Provides visualization of the output of the C-layer for each feature map.

✅ Allows saving and loading the trained network using Python's pickle module.

## 📥 Installation

Ensure you have Python 3.8+ installed. Install the required dependencies using:

pip install -r requirements.txt

## 🚀 Usage

▶️ Running the Neocognitron

To train and test the Neocognitron, simply run:

python run.py

## ⚙️ Modifying Network Parameters

To modify the network structure and parameters, edit the initStructure.py file before running the program.

## 📦 Dependencies

📌 numpy

📌 matplotlib

📌 pickle (built-in Python module)

## 📚 References

📖 Fukushima, K. "Neocognitron: A Self-organizing Neural Network Model for a Mechanism of Pattern Recognition Unaffected by Shift in Position." Biological Cybernetics, 1980.

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.

