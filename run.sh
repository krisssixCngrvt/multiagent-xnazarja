#!/bin/bash

# Build and run the Multi-Agent Foraging RL project

echo "Building Multi-Agent Foraging RL Coach..."

# Create necessary directories
mkdir -p models
mkdir -p logs

# Build the project with Maven
mvn clean package -DskipTests

if [ $? -eq 0 ]; then
    echo "Build successful!"
    echo ""
    echo "Choose agent type:"
    echo "1) Q-Learning"
    echo "2) Deep Q-Network (DQN)"
    read -p "Enter choice [1-2]: " choice
    
    case $choice in
        1)
            echo "Running with Q-Learning agents..."
            java -jar target/multiagent-foraging-rl-1.0-SNAPSHOT.jar qlearning 4 500
            ;;
        2)
            echo "Running with DQN agents..."
            java -Xmx4g -jar target/multiagent-foraging-rl-1.0-SNAPSHOT.jar dqn 4 500
            ;;
        *)
            echo "Invalid choice. Running with Q-Learning (default - faster)..."
            java -jar target/multiagent-foraging-rl-1.0-SNAPSHOT.jar qlearning 4 500
            ;;
    esac
else
    echo "Build failed!"
    exit 1
fi
