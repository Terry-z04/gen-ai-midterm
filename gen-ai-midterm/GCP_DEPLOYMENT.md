# GCP Compute Engine Deployment Guide
## UChicago MS-ADS RAG System

---

## 🚀 **OVERVIEW**

This guide will help you deploy the RAG system on Google Cloud Platform (GCP) Compute Engine.

**What you'll deploy:**
- Flask web application
- Advanced RAG system (HyDE + RAG Fusion)
- ChromaDB vector database
- OpenAI GPT integration
- Production-ready with Gunicorn

---

## 📋 **PREREQUISITES**

1. Google Cloud Platform account
2. GCP Project created
3. Billing enabled
4. OpenAI API Key
5. Firecrawl API Key
6. LangSmith API Key

---

## 🖥️ **STEP 1: CREATE COMPUTE ENGINE INSTANCE**

### **Option A: Using GCP Console**

1. Go to [GCP Console](https://console.cloud.google.com/)
2. Navigate to **Compute Engine > VM instances**
3. Click **Create Instance**

### **Recommended Configuration:**

**Instance Details:**
- **Name:** `uchicago-msads-rag`
- **Region:** `us-central1` (or closest to you)
- **Zone:** `us-central1-a`

**Machine Configuration:**
- **Series:** E2 or N2
- **Machine type:** 
  - `e2-standard-4` (4 vCPU, 16 GB RAM) - Recommended
  - `e2-standard-2` (2 vCPU, 8 GB RAM) - Minimum
  - `n2-standard-4` (4 vCPU, 16 GB RAM) - High performance

**Boot Disk:**
- **Operating System:** Ubuntu
- **Version:** Ubuntu 22.04 LTS
- **Boot disk type:** Balanced persistent disk
- **Size:** 50 GB

**Firewall:**
- ✅ Allow HTTP traffic
- ✅ Allow HTTPS traffic

**Networking:**
- **External IP:** Ephemeral (or reserve static IP for production)

### **Option B: Using gcloud CLI**

```bash
# Set project
gcloud config set project YOUR_PROJECT_ID

# Create instance
gcloud compute instances create uchicago-msads-rag \
  --zone=us-central1-a \
  --machine-type=e2-standard-4 \
  --image-family=ubuntu-2204-lts \
  --image-project=ubuntu-os-cloud \
  --boot-disk-size=50GB \
  --boot-disk-type=pd-balanced \
  --tags=http-server,https-server

# Create firewall rules (if not exists)
gcloud compute firewall-rules create allow-http \
  --allow tcp:80 \
  --target-tags http-server \
  --description="Allow HTTP traffic"

gcloud compute firewall-rules create allow-https \
  --allow tcp:443 \
  --target-tags https-server \
  --description="Allow HTTPS traffic"
```

---

## 🔧 **STEP 2: CONNECT TO YOUR INSTANCE**

### **Option A: SSH from Console**
1. Go to **Compute Engine > VM instances**
2. Click **SSH** button next to your instance

### **Option B: SSH from gcloud CLI**
```bash
gcloud compute ssh uchicago-msads-rag --zone=us-central1-a
```

### **Option C: SSH with custom key**
```bash
# Generate SSH key (if needed)
ssh-keygen -t rsa -f ~/.ssh/gcp-rag-key -C your-email@example.com

# Add to instance
gcloud compute instances add-metadata uchicago-msads-rag \
  --zone=us-central1-a \
  --metadata-from-file ssh-keys=~/.ssh/gcp-rag-key.pub

# Connect
ssh -i ~/.ssh/gcp-rag-key your-email@EXTERNAL_IP
```

---

## 📦 **STEP 3: INSTALL DEPENDENCIES**

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Python 3.10+
sudo apt install python3-pip python3-venv -y

# Install system dependencies
sudo apt install build-essential python3-dev -y

# Install nginx (for reverse proxy)
sudo apt install nginx -y

# Install git
sudo apt install git -y

# Install monitoring tools
sudo apt install htop -y
```

---

## 📁 **STEP 4: UPLOAD YOUR PROJECT**

### **Option A: Using gcloud scp**
```bash
# From your local machine
cd "/path/to/your/project"
gcloud compute scp --recurse gen-ai-midterm uchicago-msads-rag:~/ --zone=us-central1-a
```

### **Option B: Using Git**
```bash
# On the VM
git clone your-repo-url
cd gen-ai-midterm
```

### **Option C: Using Cloud Storage**
```bash
# From local machine - upload to bucket
gsutil -m cp -r gen-ai-midterm gs://your-bucket-name/

# On VM - download from bucket
gsutil -m cp -r gs://your-bucket-name/gen-ai-midterm ~/
```

---

## 🐍 **STEP 5: SETUP PYTHON ENVIRONMENT**

```bash
# Navigate to project
cd ~/gen-ai-midterm

# Create virtual environment
python3 -m venv venv

# Activate environment
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install requirements
pip install -r requirements.txt
```

---

## 🔑 **STEP 6: CONFIGURE ENVIRONMENT VARIABLES**

```bash
# Create .env file
nano .env
```

Add the following:
```bash
# OpenAI
OPENAI_API_KEY=your_openai_api_key_here

# Firecrawl (optional)
FIRECRAWL_API_KEY=your_firecrawl_api_key_here

# LangSmith (optional)
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your_langsmith_api_key_here
LANGCHAIN_PROJECT=uchicago-msads-rag

# Flask
SECRET_KEY=your_secret_key_here_use_strong_password
FLASK_ENV=production
```

Save and exit (Ctrl+X, Y, Enter)

```bash
# Secure the file
chmod 600 .env
```

### **Alternative: Using GCP Secret Manager**

```bash
# Store secrets in Secret Manager
echo -n "your_openai_key" | gcloud secrets create openai-api-key --data-file=-
echo -n "your_firecrawl_key" | gcloud secrets create firecrawl-api-key --data-file=-

# Grant access to compute engine service account
PROJECT_NUMBER=$(gcloud projects describe YOUR_PROJECT_ID --format="value(projectNumber)")
gcloud secrets add-iam-policy-binding openai-api-key \
  --member="serviceAccount:${PROJECT_NUMBER}-compute@developer.gserviceaccount.com" \
  --role="roles/secretmanager.secretAccessor"

# Retrieve in application (modify config.py)
from google.cloud import secretmanager
client = secretmanager.SecretManagerServiceClient()
name = f"projects/YOUR_PROJECT_ID/secrets/openai-api-key/versions/latest"
response = client.access_secret_version(request={"name": name})
OPENAI_API_KEY = response.payload.data.decode("UTF-8")
```

---

## ⚙️ **STEP 7: SETUP SYSTEMD SERVICE**

```bash
# Create systemd service file
sudo nano /etc/systemd/system/rag-app.service
```

Paste the following:
```ini
[Unit]
Description=UChicago MS-ADS RAG System
After=network.target

[Service]
Type=notify
User=$USER
Group=$USER
WorkingDirectory=/home/$USER/gen-ai-midterm
Environment="PATH=/home/$USER/gen-ai-midterm/venv/bin"
ExecStart=/home/$USER/gen-ai-midterm/venv/bin/gunicorn --config gunicorn_config.py app:app
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Replace `$USER` with your username (run `whoami` to check).

```bash
# Reload systemd
sudo systemctl daemon-reload

# Enable service
sudo systemctl enable rag-app

# Start service
sudo systemctl start rag-app

# Check status
sudo systemctl status rag-app
```

---

## 🌐 **STEP 8: CONFIGURE NGINX REVERSE PROXY**

```bash
# Create nginx config
sudo nano /etc/nginx/sites-available/rag-app
```

Paste the following:
```nginx
server {
    listen 80;
    server_name _;  # Replace with your domain or leave as is

    location / {
        proxy_pass http://127.0.0.1:5000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_cache_bypass $http_upgrade;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_read_timeout 120s;
        proxy_connect_timeout 120s;
    }

    location /api/ {
        proxy_pass http://127.0.0.1:5000/api/;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_cache_bypass $http_upgrade;
        proxy_read_timeout 120s;
        proxy_connect_timeout 120s;
    }
}
```

```bash
# Enable site
sudo ln -s /etc/nginx/sites-available/rag-app /etc/nginx/sites-enabled/

# Remove default site
sudo rm /etc/nginx/sites-enabled/default

# Test configuration
sudo nginx -t

# Restart nginx
sudo systemctl restart nginx
```

---

## 🔒 **STEP 9: SETUP SSL WITH GOOGLE-MANAGED CERTIFICATES (OPTIONAL)**

### **Option A: Using Let's Encrypt (Recommended for custom domains)**

```bash
# Install certbot
sudo apt install certbot python3-certbot-nginx -y

# Get certificate (replace with your domain)
sudo certbot --nginx -d your-domain.com

# Auto-renewal is setup automatically
# Test renewal
sudo certbot renew --dry-run
```

### **Option B: Using Google-Managed SSL (for Load Balancer)**

If using Cloud Load Balancer:
1. Go to **Network Services > Load Balancing**
2. Create HTTPS load balancer
3. Add your instance as backend
4. Configure SSL certificate (Google-managed)

---

## 🌍 **STEP 10: CONFIGURE STATIC IP (OPTIONAL)**

```bash
# Reserve static external IP
gcloud compute addresses create rag-app-ip \
  --region=us-central1

# Get the IP address
gcloud compute addresses describe rag-app-ip \
  --region=us-central1 \
  --format="get(address)"

# Assign to instance
gcloud compute instances delete-access-config uchicago-msads-rag \
  --zone=us-central1-a \
  --access-config-name="external-nat"

gcloud compute instances add-access-config uchicago-msads-rag \
  --zone=us-central1-a \
  --access-config-name="external-nat" \
  --address=RAG-APP-IP
```

---

## ✅ **STEP 11: VERIFY DEPLOYMENT**

```bash
# Check app status
sudo systemctl status rag-app

# Check nginx status
sudo systemctl status nginx

# View app logs
sudo journalctl -u rag-app -f

# Test health endpoint
curl http://localhost:5000/health

# Get external IP
gcloud compute instances describe uchicago-msads-rag \
  --zone=us-central1-a \
  --format="get(networkInterfaces[0].accessConfigs[0].natIP)"
```

Access your app:
- **HTTP:** `http://EXTERNAL_IP`
- **HTTPS:** `https://your-domain.com` (if configured)

---

## 📊 **MONITORING WITH GOOGLE CLOUD**

### **Cloud Monitoring (Stackdriver)**

```bash
# Install monitoring agent
curl -sSO https://dl.google.com/cloudagents/add-google-cloud-ops-agent-repo.sh
sudo bash add-google-cloud-ops-agent-repo.sh --also-install

# Configure monitoring
sudo service google-cloud-ops-agent restart
```

### **View Logs**
1. Go to **Logging > Logs Explorer**
2. Filter by resource: `gce_instance`
3. Search for your logs

### **Set Up Alerts**
1. Go to **Monitoring > Alerting**
2. Create alert policy
3. Set conditions (CPU, Memory, etc.)

---

## 🔄 **UPDATING YOUR APP**

```bash
# SSH to instance
gcloud compute ssh uchicago-msads-rag --zone=us-central1-a

# Navigate to project
cd ~/gen-ai-midterm

# Activate environment
source venv/bin/activate

# Pull latest changes or upload new files
git pull  # if using git
# or
# gcloud compute scp --recurse gen-ai-midterm uchicago-msads-rag:~/ --zone=us-central1-a

# Install new dependencies
pip install -r requirements.txt

# Restart service
sudo systemctl restart rag-app

# Check status
sudo systemctl status rag-app
```

---

## 💰 **COST OPTIMIZATION**

### **GCP Compute Engine Costs (us-central1):**
- e2-standard-2: ~$48/month
- e2-standard-4: ~$97/month
- n2-standard-4: ~$150/month

### **Cost-Saving Tips:**

1. **Use Committed Use Discounts (CUDs):**
   - Save up to 57% with 1-year commitment
   - Save up to 70% with 3-year commitment

2. **Use Sustained Use Discounts:**
   - Automatic discounts for running instances 25%+ of month

3. **Use Preemptible/Spot VMs:**
   - Up to 80% cheaper
   - Good for development/testing

4. **Stop instances when not in use:**
   ```bash
   gcloud compute instances stop uchicago-msads-rag --zone=us-central1-a
   gcloud compute instances start uchicago-msads-rag --zone=us-central1-a
   ```

5. **Use Cloud Storage for data:**
   - Cheaper than persistent disks for large datasets

---

## 🐛 **TROUBLESHOOTING**

### **App won't start:**
```bash
# Check logs
sudo journalctl -u rag-app -n 100 --no-pager

# Check permissions
ls -la ~/gen-ai-midterm

# Check Python
~/gen-ai-midterm/venv/bin/python --version

# Manually run to see errors
cd ~/gen-ai-midterm
source venv/bin/activate
python app.py
```

### **502 Bad Gateway:**
```bash
# Check if app is running
sudo systemctl status rag-app

# Check nginx
sudo nginx -t
sudo systemctl status nginx

# Restart both
sudo systemctl restart rag-app
sudo systemctl restart nginx
```

### **Can't connect externally:**
```bash
# Check firewall rules
gcloud compute firewall-rules list

# Check if instance has external IP
gcloud compute instances describe uchicago-msads-rag \
  --zone=us-central1-a \
  --format="get(networkInterfaces[0].accessConfigs[0].natIP)"

# Test from instance
curl http://localhost:5000/health
```

### **Out of Memory:**
```bash
# Add swap
sudo fallocate -l 4G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab

# Or upgrade machine type
gcloud compute instances stop uchicago-msads-rag --zone=us-central1-a
gcloud compute instances set-machine-type uchicago-msads-rag \
  --machine-type=e2-standard-4 \
  --zone=us-central1-a
gcloud compute instances start uchicago-msads-rag --zone=us-central1-a
```

---

## 🔐 **SECURITY BEST PRACTICES**

1. **Use IAM roles properly:**
   ```bash
   # Create service account
   gcloud iam service-accounts create rag-app-sa \
     --display-name="RAG App Service Account"
   
   # Assign minimal permissions
   ```

2. **Enable OS Login:**
   ```bash
   gcloud compute project-info add-metadata \
     --metadata enable-oslogin=TRUE
   ```

3. **Use VPC firewall rules:**
   ```bash
   # Restrict SSH to your IP
   gcloud compute firewall-rules create allow-ssh-from-my-ip \
     --allow tcp:22 \
     --source-ranges YOUR_IP/32 \
     --target-tags uchicago-msads-rag
   ```

4. **Regular updates:**
   ```bash
   sudo apt update && sudo apt upgrade -y
   ```

5. **Use Secret Manager for API keys** (as shown in Step 6)

---

## 📈 **SCALING OPTIONS**

### **Vertical Scaling:**
```bash
# Stop instance
gcloud compute instances stop uchicago-msads-rag --zone=us-central1-a

# Change machine type
gcloud compute instances set-machine-type uchicago-msads-rag \
  --machine-type=e2-standard-8 \
  --zone=us-central1-a

# Start instance
gcloud compute instances start uchicago-msads-rag --zone=us-central1-a
```

### **Horizontal Scaling:**
1. Create instance template
2. Set up managed instance group
3. Configure autoscaling
4. Add load balancer

```bash
# Create instance template
gcloud compute instance-templates create rag-app-template \
  --machine-type=e2-standard-4 \
  --image-family=ubuntu-2204-lts \
  --image-project=ubuntu-os-cloud \
  --boot-disk-size=50GB

# Create managed instance group
gcloud compute instance-groups managed create rag-app-group \
  --base-instance-name=rag-app \
  --template=rag-app-template \
  --size=2 \
  --zone=us-central1-a

# Configure autoscaling
gcloud compute instance-groups managed set-autoscaling rag-app-group \
  --max-num-replicas=10 \
  --min-num-replicas=2 \
  --target-cpu-utilization=0.7 \
  --zone=us-central1-a
```

---

## ✅ **QUICK START CHECKLIST**

- [ ] Create GCP project & enable billing
- [ ] Create Compute Engine instance (e2-standard-4)
- [ ] Configure firewall rules
- [ ] SSH to instance
- [ ] Install dependencies (Python, nginx, etc.)
- [ ] Upload project files
- [ ] Create virtual environment
- [ ] Install Python packages
- [ ] Configure .env file
- [ ] Setup systemd service
- [ ] Configure Nginx
- [ ] Setup SSL (optional)
- [ ] Reserve static IP (optional)
- [ ] Test deployment
- [ ] Setup monitoring

---

## 🎉 **YOU'RE LIVE ON GCP!**

Your RAG system is now running on Google Cloud Platform!

**Access your app:**
- Web Interface: `http://EXTERNAL_IP`
- API Health: `http://EXTERNAL_IP/health`
- API Endpoint: `http://EXTERNAL_IP/api/query`

**Management:**
```bash
# SSH
gcloud compute ssh uchicago-msads-rag --zone=us-central1-a

# Check logs
sudo journalctl -u rag-app -f

# Restart app
sudo systemctl restart rag-app

# Monitor resources
htop

# Stop instance (save costs)
gcloud compute instances stop uchicago-msads-rag --zone=us-central1-a

# Start instance
gcloud compute instances start uchicago-msads-rag --zone=us-central1-a
```

---

## 📚 **ADDITIONAL RESOURCES**

- [GCP Compute Engine Docs](https://cloud.google.com/compute/docs)
- [GCP Pricing Calculator](https://cloud.google.com/products/calculator)
- [GCP Free Tier](https://cloud.google.com/free)
- [Cloud Monitoring](https://cloud.google.com/monitoring/docs)
- [Secret Manager](https://cloud.google.com/secret-manager/docs)
