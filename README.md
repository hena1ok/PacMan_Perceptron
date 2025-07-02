# 🧠 PacMan Perceptron

A self-playing **Pac-Man agent** powered by a **single-layer perceptron**.  
This AI learns how to play Pac-Man through simple linear classification based on in-game states and features.

---

## 🎯 Project Overview

This project implements a **perceptron-based learning agent** for Pac-Man.  
The perceptron classifies possible actions (e.g., move left, right, up, down) based on game state features and chooses the one with the highest score.

---

## 🧠 AI Technique Used

- **Single-Layer Perceptron**
- **Supervised Learning** from past games or expert decisions
- **Feature-Based Decision Making** from game state inputs

---

## 🛠️ Tech Stack

- **Python 3**
- Basic libraries (`math`, `random`)
- No external ML libraries required

---

## 🚀 How to Run

1. Clone the repository:

```bash
git clone https://github.com/hena1ok/PacMan_Perceptron.git
cd PacMan_Perceptron
```

2. Run the main file:

```bash
python main.py
```

3. Follow on-screen instructions to observe the AI in action.

---

## 📦 Project Structure

```
PacMan_Perceptron/
│
├── main.py           # Main simulation and game loop
├── perceptron.py     # Core perceptron implementation (if separated)
├── agent.py          # Pac-Man AI agent logic
├── utils.py          # Helper functions and game environment
└── README.md
```

---

## 📘 Key Features

- Trains a Pac-Man agent to play using simple weight updates.
- Dynamically selects actions based on the current environment state.
- No need for deep learning frameworks.

---

## 📈 Potential Improvements

- Add multi-layer perceptrons (MLPs)
- Use reinforcement learning (Q-learning, DQN)
- Visualize agent performance over time

---

## 👨‍💻 Author

Made with ❤️ by Henok Yizelkal  
GitHub: [@hena1ok](https://github.com/hena1ok)

---

## 📄 License

This project is open-source and free to use for academic and educational purposes.
