COMPOSE := docker compose
SERVICE := app

.PHONY: build up down

build:
	$(COMPOSE) build $(SERVICE)

up:
	$(COMPOSE) up -d $(SERVICE)

down:
	$(COMPOSE) down
