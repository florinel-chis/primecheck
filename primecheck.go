package main

import (
	"context"
	"flag"
	"fmt"
	"os"
	"strconv"
	"strings"
	"time"

	"github.com/joho/godotenv"
	"github.com/openai/openai-go"
	"github.com/openai/openai-go/option"
)

// Logger is a simple logging utility
type Logger struct {
	verbose bool
	prefix  string
}

// NewLogger creates a new logger with the given verbosity
func NewLogger(verbose bool) *Logger {
	return &Logger{
		verbose: verbose,
		prefix:  "[INFO] ",
	}
}

// Log prints a message if verbose mode is enabled
func (l *Logger) Log(format string, a ...interface{}) {
	if l.verbose {
		message := fmt.Sprintf(format, a...)
		timestamp := time.Now().Format("2006-01-02 15:04:05.000")
		fmt.Fprintf(os.Stderr, "%s %s%s\n", timestamp, l.prefix, message)
	}
}

// Error logs an error message (always printed)
func (l *Logger) Error(format string, a ...interface{}) {
	oldPrefix := l.prefix
	l.prefix = "[ERROR] "
	message := fmt.Sprintf(format, a...)
	timestamp := time.Now().Format("2006-01-02 15:04:05.000")
	fmt.Fprintf(os.Stderr, "%s %s%s\n", timestamp, l.prefix, message)
	l.prefix = oldPrefix
}

// Debug logs a debug message
func (l *Logger) Debug(format string, a ...interface{}) {
	oldPrefix := l.prefix
	l.prefix = "[DEBUG] "
	l.Log(format, a...)
	l.prefix = oldPrefix
}

func main() {
	// Define command-line flags
	verboseFlag := flag.Bool("v", false, "Enable verbose logging")
	flag.Parse()

	// Initialize logger
	logger := NewLogger(*verboseFlag)
	logger.Log("Starting prime-checker application")

	startTime := time.Now()

	// Load .env file
	logger.Log("Attempting to load .env file")
	err := godotenv.Load()
	if err != nil {
		logger.Log("Warning: .env file not found or could not be loaded")
		fmt.Println("Warning: .env file not found or could not be loaded")
		// Continue execution as the API key might be set directly in the environment
	} else {
		logger.Log(".env file loaded successfully")
	}

	// Check if API key is set
	logger.Log("Checking for OpenAI API key in environment")
	apiKey, exists := os.LookupEnv("OPENAI_API_KEY")
	if !exists || apiKey == "" {
		logger.Error("OPENAI_API_KEY environment variable not set")
		fmt.Println("Error: OPENAI_API_KEY environment variable not set")
		fmt.Println("Please set it in your .env file or in your environment")
		os.Exit(1)
	}
	logger.Log("OpenAI API key found in environment")

	// Get timeout from .env or use default
	logger.Log("Checking for timeout configuration")
	timeoutStr := os.Getenv("MAX_TIMEOUT_SECONDS")
	timeoutSeconds := 30 // Default timeout
	if timeoutStr != "" {
		logger.Log("Found MAX_TIMEOUT_SECONDS: %s", timeoutStr)
		if parsedTimeout, err := strconv.Atoi(timeoutStr); err == nil && parsedTimeout > 0 {
			timeoutSeconds = parsedTimeout
			logger.Log("Using configured timeout: %d seconds", timeoutSeconds)
		} else {
			logger.Log("Invalid timeout value, using default: %d seconds", timeoutSeconds)
		}
	} else {
		logger.Log("No timeout configured, using default: %d seconds", timeoutSeconds)
	}

	// Process command-line arguments (excluding our flags)
	args := flag.Args()
	logger.Log("Command-line arguments: %v", args)

	// Validate CLI arguments
	if len(args) != 1 {
		logger.Error("Invalid number of arguments: expected 1, got %d", len(args))
		fmt.Println("Usage: prime-checker <number>")
		os.Exit(1)
	}

	// Parse the number
	numStr := args[0]
	logger.Log("Parsing number: %s", numStr)
	num, err := strconv.ParseInt(numStr, 10, 64)
	if err != nil {
		logger.Error("Failed to parse number '%s': %v", numStr, err)
		fmt.Printf("Error: Invalid number '%s'\n", numStr)
		os.Exit(1)
	}
	logger.Log("Successfully parsed number: %d", num)

	// Check if the number is prime using ChatGPT
	logger.Log("Checking if %d is prime using OpenAI API", num)
	result, err := checkPrime(logger, apiKey, num, timeoutSeconds)
	if err != nil {
		logger.Error("Prime check failed: %v", err)
		fmt.Printf("Error: %v\n", err)
		os.Exit(1)
	}

	// Output only yes or no
	logger.Log("Prime check result: %s", result)
	fmt.Println(result)

	// Log execution time
	elapsedTime := time.Since(startTime)
	logger.Log("Total execution time: %v", elapsedTime)
}

func checkPrime(logger *Logger, apiKey string, number int64, timeoutSeconds int) (string, error) {
	logger.Log("Initializing OpenAI client")
	// Initialize the official OpenAI client
	client := openai.NewClient(
		option.WithAPIKey(apiKey),
	)

	// Create a context with timeout
	logger.Log("Creating context with timeout: %d seconds", timeoutSeconds)
	ctx, cancel := context.WithTimeout(context.Background(), time.Duration(timeoutSeconds)*time.Second)
	defer cancel()

	// Prepare the prompt message with best practices
	logger.Log("Preparing prompt for number: %d", number)
	prompt := fmt.Sprintf(`You are solving a mathematical problem that requires a one-word answer.

TASK: Determine if %d is a prime number.

DEFINITION: A prime number is a natural number greater than 1 that is not a product of two smaller natural numbers.

REQUIREMENTS:
- Answer with ONLY the word "yes" or "no"
- "yes" if %d is prime
- "no" if %d is not prime
- Do not include explanations, periods, or any other text

Output:`, number, number, number)

	logger.Debug("Using prompt: %s", prompt)

	// Create the API request
	logger.Log("Sending request to OpenAI API (model: %s, maxTokens: %d, temperature: %f)",
		openai.ChatModelGPT3_5Turbo, 5, 0.0)
	requestStartTime := time.Now()

	chatCompletion, err := client.Chat.Completions.New(ctx, openai.ChatCompletionNewParams{
		Messages: []openai.ChatCompletionMessageParamUnion{
			openai.UserMessage(prompt),
		},
		Model:       openai.ChatModelGPT3_5Turbo,
		MaxTokens:   openai.Int(5),     // Even shorter since we're enforcing a one-word answer
		Temperature: openai.Float(0.0), // Set to 0 for deterministic answers
	})

	requestDuration := time.Since(requestStartTime)
	logger.Log("OpenAI API request completed in %v", requestDuration)

	if err != nil {
		logger.Error("OpenAI API request failed: %v", err)
		return "", fmt.Errorf("OpenAI API error: %v", err)
	}

	// Extract the response content
	answer := strings.TrimSpace(chatCompletion.Choices[0].Message.Content)
	logger.Log("Raw response from OpenAI: '%s'", answer)

	// Log usage information
	logger.Log("Token usage - Prompt: %d, Completion: %d, Total: %d",
		chatCompletion.Usage.PromptTokens,
		chatCompletion.Usage.CompletionTokens,
		chatCompletion.Usage.TotalTokens)

	// Sanitize to ensure we only return "yes" or "no"
	answer = strings.ToLower(answer)
	logger.Log("Normalized response: '%s'", answer)

	if answer == "yes" {
		logger.Log("Confirmed: %d is prime", number)
		return "yes", nil
	} else if answer == "no" {
		logger.Log("Confirmed: %d is not prime", number)
		return "no", nil
	}

	// If we got here, the response wasn't a clear yes or no
	logger.Error("Unexpected response format from OpenAI: '%s'", answer)
	return "", fmt.Errorf("Unexpected response from OpenAI: %s", answer)
}
