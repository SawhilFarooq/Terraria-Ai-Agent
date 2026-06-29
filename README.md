# Autonomous Terraria AI Agent

An autonomous AI agent that learns to defeat Terraria bosses using computer vision and imitation learning — with no access to game memory or APIs, only raw screen pixels.

## How It Works

The agent perceives the game entirely through screen capture, using OpenCV to detect the boss position, track health states, and parse HUD elements in real time. A convolutional neural network trained on expert gameplay recordings then decides which action to take every frame.

## Features

- **Imitation Learning Pipeline** — records expert gameplay and trains a CNN to replicate combat decisions
- **Real-Time Perception** — screen capture and OpenCV-based object detection running at 20 FPS
- **8-Action Combat Framework** — shooting, strafing, jumping, flying, dashing, and healing
- **Multi-Phase Boss Handling** — dynamic health bar recalibration when bosses enter new phases
- **Checkpoint Saving** — training progress persists across sessions

## Tech Stack

- Python
- PyTorch
- OpenCV
- NumPy
- pydirectinput
- mss

## Project Structure
```
terraria-ai-agent/
├── record.py            # Records screen and inputs during gameplay
├── train.py             # Trains CNN on recorded gameplay
├── ai_agent.py          # Runs the imitation learning agent
├── terraria_agent.py    # Rule-based agent (initial prototype)
└── check_recording.py   # Verifies recording data quality
```
## Pipeline
1. Run `record.py` while playing — captures screen frames and labeled inputs
2. Run `train.py` — trains a CNN on the recordings (get at least 5 recordings/boss-fights so that the AI has more to train off of. The more recordings the better)
3. Run `ai_agent.py` — deploys the trained agent to play autonomously
