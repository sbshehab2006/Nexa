package api

import "net/http"

func ListModelsHandler(w http.ResponseWriter, r *http.Request) {
	// Return list of models
	w.WriteHeader(http.StatusOK)
}
