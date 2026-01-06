package cmd

import (
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strings"
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
		downloadModel(modelName)
	},
}

func init() {
	rootCmd.AddCommand(pullCmd)
}

// WriteCounter counts the number of bytes written to it. It implements to the io.Writer interface
// and we can pass this into io.TeeReader() which will report progress on each write cycle.
type WriteCounter struct {
	Total      uint64
	Downloaded uint64
}

func (wc *WriteCounter) Write(p []byte) (int, error) {
	n := len(p)
	wc.Downloaded += uint64(n)
	wc.PrintProgress()
	return n, nil
}

func (wc *WriteCounter) PrintProgress() {
	// Calculate output
	const dataLength = 50
	percent := float64(wc.Downloaded) / float64(wc.Total) * 100
	
	// Create progress bar
	filledLength := int(percent / 100 * dataLength)
	bar := strings.Repeat("=", filledLength) + strings.Repeat("-", dataLength-filledLength)

	// Print to console
	// \r overwrites the current line
	fmt.Printf("\rDownloading %s [%s] %.2f%%", formatBytes(wc.Downloaded), bar, percent)
}

func formatBytes(b uint64) string {
	const unit = 1024
	if b < unit {
		return fmt.Sprintf("%d B", b)
	}
	div, exp := int64(unit), 0
	for n := b / unit; n >= unit; n /= unit {
		div *= unit
		exp++
	}
	return fmt.Sprintf("%.1f %cB", float64(b)/float64(div), "KMGTPE"[exp])
}

func downloadModel(name string) {
	// Model Registry (Mapping names to URLs)
	modelRegistry := map[string]string{
		"tinyllama": "https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
		"llama2":    "https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/resolve/main/llama-2-7b-chat.Q4_K_M.gguf",
		"mistral":   "https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF/resolve/main/mistral-7b-instruct-v0.1.Q4_K_M.gguf",
	}

	url, exists := modelRegistry[name]
	if !exists {
		// Default to TinyLlama if unknown, or error out
		fmt.Printf("Model '%s' not found in registry. Trying generic download...\n", name)
		// For demo safety, let's just error unless exact match to avoid huge accidental downloads
		fmt.Printf("Available models: tinyllama, llama2, mistral\n")
		return
	}

	destDir := "models"
	fileName := name + ".gguf"
	destPath := filepath.Join(destDir, fileName)

	if _, err := os.Stat(destDir); os.IsNotExist(err) {
		os.Mkdir(destDir, 0755)
	}

	fmt.Printf("Pulling manifest for %s...\n", name)
	
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

	// Initialize progress bar
	counter := &WriteCounter{Total: uint64(resp.ContentLength)}

    // Write the body to file with progress
	if _, err = io.Copy(out, io.TeeReader(resp.Body, counter)); err != nil {
		fmt.Printf("\nError processing download: %v\n", err)
        return
    }

	fmt.Printf("\nSuccess! Downloaded %s to %s\n", fileName, destPath)
}
