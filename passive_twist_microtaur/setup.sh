#!/bin/bash
# setup.sh - Setup environment

echo "Setting up Passive Twist MicroTaur RL project..."

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install mujoco gymnasium torch numpy scipy matplotlib tensorboard pyyaml

# Test installation
echo "Testing MuJoCo installation..."
python -c "import mujoco; print('MuJoCo version:', mujoco.__version__)"

echo ""
echo "Setup complete!"
echo "To activate environment: source venv/bin/activate"
echo "To test robot: python scripts/test_robot.py"
echo "To train: python training/train.py"