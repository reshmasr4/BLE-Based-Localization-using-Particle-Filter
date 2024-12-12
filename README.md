# BLE-Based Localization using Particle Filter

This repository contains the implementation of a **BLE-based localization system** utilizing **particle filters** to estimate the position of objects in a hybrid localization system. The project focuses on accurate proximity estimation using RSSI (Received Signal Strength Indicator) data for real-time object tracking and localization.

---

## Table of Contents
1. [Introduction](#introduction)
2. [Features](#features)
3. [Installation](#installation)
4. [License](#license)


---

## Introduction

Bluetooth Low Energy (BLE) is widely used for indoor positioning and proximity estimation. This project leverages a **particle filter algorithm** to enhance position accuracy in environments where BLE signal strength is used for tracking. The key focus is to refine RSSI-based proximity estimation and integrate it into a simulation framework for **herd management systems** like the *WeideInsight* project.

### Key Objectives:
- Improve proximity estimation accuracy based on RSSI values.
- Simulate particle filter behavior in hybrid localization systems.
- Use particle weighting techniques to achieve realistic proximity estimations.

---

## Features

- **Particle Filter Simulation**:
  - RSSI-based particle weighting.
  - Continuous position updates.
  
- **Static and Continuous Proximity Estimation**:
  - Simulated real-world BLE beacon scenarios.
  - Comparison of static and dynamic positioning techniques.

- **Visualization**:
  - Simulation frames for position and proximity evaluation.

- **Hybrid Localization**:
  - Integration of BLE and LPWAN data for future expansion.

---

---

## Installation

### Prerequisites
- Python 3.8 or higher
- Required Python libraries:
  - `numpy`
  - `matplotlib`
  - `pygame`
  - `pandas`

## License

This project is licensed under the MIT License. See the LICENSE file for details.
