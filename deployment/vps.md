# VPS deployment of API services (api, rabbitmq, minio, broker worker etc.. No AI workers on this server)

## 1. SSH into VPS
```bash
ssh username@YOUR_VPS_IP
```

## 2. Update Ubuntu
```bash
sudo apt update
sudo apt upgrade -y
```

## 3. Install Docker
```bash
sudo apt install -y docker.io docker-compose-plugin

docker --version
docker compose version
```

## 4. Clone the project from github
```bash
git clone YOUR_REPOSITORY_URL
cd YOUR_REPOSITORY
```

## 5. use .env_gc configuration or update environment variables with correct VPS public IP host
Then run docker compose commands with
```bash
docker compose --env-file .env_gc -f docker-compose.gc.yml ...
```

## 6. Build the docker image
```bash
docker compose --env-file .env_gc -f docker-compose.gc.yml build
```

## 7. Start the services
```bash
docker compose --env-file .env_gc -f docker-compose.gc.yml up -d api rabbitmq file_storage broker_worker
```
The APi should be accesible on http://YOUR_VPS_IP:API_PORT  (port is defined as API_PORT inside .env_gc)
http://YOUR_VPS_IP:8000/docs

!!! Here you can test API working by starting diarization service on local PC and using VPS IP and PORT as access for rabbitmq and minio

## 8. Register domain on cloudflare (or other) and then install NGINX and configure it
```bash
sudo apt install -y nginx

sudo nano /etc/nginx/sites-available/kt-api
```
Add:
```bash
server {
    listen 80;
    server_name api.yourdomain.com;

    client_max_body_size 2G;

    location / {
        proxy_pass http://127.0.0.1:API_PORT;

        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        proxy_read_timeout 300s;
        proxy_send_timeout 300s;
    }
}
```
Then:
```bash
sudo ln -s /etc/nginx/sites-available/kt-api /etc/nginx/sites-enabled/kt-api

sudo nginx -t

sudo systemctl reload nginx
```

## 9. Add HTTPS
```bash
sudo apt install -y certbot python3-certbot-nginx

sudo certbot --nginx -d api.yourdomain.com

```
-> https://api.yourdomain.com/docs

## 10. Firewall ports
```bash
sudo ufw allow OpenSSH
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
sudo ufw allow 5672/tcp
sudo ufw allow 9000/tcp
sudo ufw enable
```

In docker compose can change API ports:
```
...
api:
  ports:
    - "127.0.0.1:${API_PORT}:${API_PORT}"
...
```

