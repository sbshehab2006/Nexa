package cmd

import (
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"time"

	"github.com/spf13/cobra"
)

// pullCmd represents the pull command
var pullCmd = &cobra.Command{
	Use:   "pull [model_name]",
	Short: "Download a model from the official Nexa registry",
	Args:  cobra.ExactArgs(1),
	Run: func(cmd *cobra.Command, args []string) {
		modelName := args[0]
		fmt.Printf("Pulling model '%s'...\n", modelName)
		downloadModel(modelName)
	},
}

func init() {
	rootCmd.AddCommand(pullCmd)
}

func downloadModel(name string) {
	// For now, we simulate a download or use a dummy URL
	// In a real app, this would query a registry API
	
	url := "https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
	destDir := "models"
	fileName := name + ".gguf"
	destPath := filepath.Join(destDir, fileName)

	if _, err := os.Stat(destDir); os.IsNotExist(err) {
		os.Mkdir(destDir, 0755)
	}

	fmt.Printf("Downloading to %s...\n", destPath)
	
	// Create the file
    out, err := os.Create(destPath)
    if err != nil {
		fmt.Printf("Error creating file: %v\n", err)
        return
    }
    defer out.Close()

    // Get the data
    resp, err := http.Get(url)
    if err != nil {
		fmt.Printf("Error downloading: %v\n", err)
        return
    }
    defer resp.Body.Close()

    // Write the body to file
	// Using a simple copy for now. For progress bars, we'd need more logic.
	
	start := time.Now()
    _, err = io.Copy(out, resp.Body)
    if err != nil {
		fmt.Printf("Error processing download: %v\n", err)
        return
    }

	elapsed := time.Since(start)
    fmt.Printf("Success! Downloaded %s in %s\n", fileName, elapsed)
}
