package rl;

/**
 * Main entry point for the Multi-Agent Foraging RL system.
 * Optimized for achieving the highest possible score.
 */
public class Main {
    
    public static void main(String[] args) {
        System.out.println("╔══════════════════════════════════════════════════════════╗");
        System.out.println("║     Multi-Agent Foraging Reinforcement Learning Coach    ║");
        System.out.println("║              Optimized for Maximum Score                 ║");
        System.out.println("╚══════════════════════════════════════════════════════════╝\n");
        
        // Parse command line arguments
        int numAgents = getIntArg(args, "--agents", 3);
        int gridWidth = getIntArg(args, "--width", 10);
        int gridHeight = getIntArg(args, "--height", 10);
        int numFood = getIntArg(args, "--food", 15);
        int numObstacles = getIntArg(args, "--obstacles", 5);
        int maxSteps = getIntArg(args, "--max-steps", 200);
        int trainEpisodes = getIntArg(args, "--train-episodes", 2000);
        int evalEpisodes = getIntArg(args, "--eval-episodes", 10);
        boolean demo = hasArg(args, "--demo");
        boolean quickRun = hasArg(args, "--quick");
        
        // Quick run for testing
        if (quickRun) {
            trainEpisodes = 500;
            evalEpisodes = 5;
        }
        
        // Create coach with optimized settings
        MultiAgentCoach coach = new MultiAgentCoach(
            numAgents, gridWidth, gridHeight, 
            numFood, numObstacles, maxSteps
        );
        
        // Train the agents
        long startTime = System.currentTimeMillis();
        coach.train(trainEpisodes, true);
        long trainTime = System.currentTimeMillis() - startTime;
        
        System.out.printf("\nTraining completed in %.2f seconds%n", trainTime / 1000.0);
        
        // Evaluate performance
        double avgScore = coach.evaluate(evalEpisodes);
        
        // Show demonstration if requested
        if (demo) {
            coach.demonstrate();
        }
        
        // Print final summary
        System.out.println("\n╔══════════════════════════════════════════════════════════╗");
        System.out.println("║                    FINAL RESULTS                         ║");
        System.out.println("╠══════════════════════════════════════════════════════════╣");
        System.out.printf("║  Best Training Score:     %8.2f                       ║%n", coach.getBestScore());
        System.out.printf("║  Average Evaluation Score: %8.2f                       ║%n", avgScore);
        System.out.printf("║  Training Time:           %8.2f seconds               ║%n", trainTime / 1000.0);
        System.out.println("╚══════════════════════════════════════════════════════════╝");
        
        // Exit with score indicator
        if (avgScore >= 100) {
            System.out.println("\n🏆 EXCELLENT! Maximum score achieved!");
            System.exit(0);
        } else if (avgScore >= 80) {
            System.out.println("\n⭐ GREAT! High score achieved!");
            System.exit(0);
        } else if (avgScore >= 60) {
            System.out.println("\n✓ GOOD! Decent score achieved!");
            System.exit(0);
        } else {
            System.out.println("\n→ Training may need more episodes for better results.");
            System.exit(0);
        }
    }
    
    private static int getIntArg(String[] args, String name, int defaultValue) {
        for (int i = 0; i < args.length - 1; i++) {
            if (args[i].equals(name)) {
                try {
                    return Integer.parseInt(args[i + 1]);
                } catch (NumberFormatException e) {
                    return defaultValue;
                }
            }
        }
        return defaultValue;
    }
    
    private static boolean hasArg(String[] args, String name) {
        for (String arg : args) {
            if (arg.equals(name)) {
                return true;
            }
        }
        return false;
    }
}
