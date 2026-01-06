package server

import (
	"log"
	"net/http"
)

func Serve() {
	r := NewRouter()
	log.Println("Listening on :8080")
	log.Fatal(http.ListenAndServe(":8080", r))
}
