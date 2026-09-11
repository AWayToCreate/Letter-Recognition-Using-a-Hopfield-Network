# Letter Recognition Using a Hopfield Network

A Python project implementing a **Hopfield neural network** capable of recognizing and reconstructing letters from images containing different levels of noise.

## Objective

The objective of this project is to study how a Hopfield network works and its ability to recover a memorized pattern from a noisy version of the original.

The network learns three patterns representing the letters **A, B, and C**, and then attempts to reconstruct them when a portion of their pixels has been modified.

## Technologies

* Python
* NumPy
* Pillow
* Matplotlib

## How It Works

The program follows several steps:

1. Loads the letter images in `.ppm` format
2. Resizes the images to `15 × 15` matrices
3. Converts pixel values into binary states (`-1 / 1`)
4. Builds the network's weight matrix
5. Applies the **Hebbian learning rule**
6. Adds different levels of noise to the letter patterns
7. Reconstructs the patterns using the Hopfield network
8. Calculates the network's energy
9. Measures similarity with the original pattern
10. Identifies the recognized letter

## Hopfield Network

Each pixel in the image is represented by a neuron.

With images of `15 × 15` pixels, the network therefore contains **225 neurons**.

The letters **A, B, and C** are used as patterns stored in the network. When a noisy pattern is presented, the neurons are progressively updated until the network converges toward a stable state corresponding to one of the learned patterns.

This allows the network to recover the original letter even when a significant portion of its pixels has been altered.

## Experimentation

The program evaluates the network using three different noise levels:

* **10%**
* **30%**
* **45%**

For each noise level, the program analyzes:

* The evolution of the network's energy
* The similarity with the original pattern
* The reconstructed letter
* The final letter recognized by the network

The results are visualized using **Matplotlib**, making it possible to observe how noise affects the network's ability to retrieve stored patterns.

## Results

**graphic of the evolution of the energy and the similarity for a letter and a noise associated**
<img width="1000" height="600" alt="image" src="https://github.com/user-attachments/assets/f75519aa-263d-4c02-b529-655aaa958fb3" />

**example of a letter being recognised by the neural network**
<img width="567" height="838" alt="image" src="https://github.com/user-attachments/assets/4947c502-1afb-4cf8-bf1f-6984742f4ae3" />



## Project Structure

```text
reseau_de_neurones/
│
├── lettres_ppm/
│   ├── A.ppm
│   ├── B.ppm
│   └── C.ppm
│
├── main.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Requirements

The project requires Python and the following libraries:

* NumPy
* Pillow
* Matplotlib

Dependencies can be installed using:

```bash
pip install -r requirements.txt
```

## Purpose

This project provides a practical implementation of **associative memory using Hopfield neural networks** and demonstrates their ability to recover stored patterns from noisy inputs.
