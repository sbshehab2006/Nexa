package server

import (
	"net/http"
	"nexa/api"
)

func NewRouter() *http.ServeMux {
	mux := http.NewServeMux()
	mux.HandleFunc("/api/chat", api.ChatHandler)
	mux.HandleFunc("/api/generate", api.GenerateHandler)
	mux.HandleFunc("/api/tags", api.ListModelsHandler)
	return mux
}
