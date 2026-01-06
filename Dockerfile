FROM golang:1.22 AS builder

WORKDIR /app
COPY . .
RUN go build -o nexa main.go

FROM debian:bookworm-slim
COPY --from=builder /app/nexa /usr/local/bin/nexa
CMD ["nexa", "serve"]
