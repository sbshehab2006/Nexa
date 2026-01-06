package fs

import "os"

func ValidateModel(path string) bool {
	_, err := os.Stat(path)
	return err == nil
}
