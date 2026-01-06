package app

import "log"

type App struct {
	Config *Config
}

func New(cfg *Config) *App {
	return &App{Config: cfg}
}

func (a *App) Start() {
	log.Println("App Started")
}
