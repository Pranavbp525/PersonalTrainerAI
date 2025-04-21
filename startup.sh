#!/bin/bash
echo "[startup] Script started at $(date)" >> ~/startup.log
# Step 1: Activate virtual environment
source env/bin/activate
echo "[startup] Virtualenv activated" >> ~/startup.log

# Step 2: Copy latest files from Cloud Storage
echo "Syncing latest files from GCS..."
gsutil -m cp -r gs://project_files_rag/* ~/
echo "[startup] Files pulled from GCS" >> ~/startup.log

# Step 3: Run your app
echo "Running app.py..."
python3 ~/app.py
echo "[startup] app.py executed" >> ~/startup.log

# Step 4: Sync preprocessed data back to GCS
echo "Uploading preprocessed data..."
gsutil -m cp -r ~/data/preprocessed_json_data gs://project_files_rag/data/

# Step 5: Sync raw JSON data back to GCS
echo "Uploading raw JSON data..."
gsutil -m cp -r ~/data/raw_json_data gs://project_files_rag/data/

echo "Uploading logs..."
gsutil -m cp -r ~/logs gs://project_files_rag/
echo "[startup] Files uploaded to GCS" >> ~/startup.log

# Step 6: Wait for a minute
echo "Waiting for 60 seconds..."
sleep 60
echo "[startup] Waited for 60 seconds" >> ~/startup.log

# Step 7: Shutdown the VM
echo "[startup] Shutting down the VM..." >> ~/startup.log
sudo shutdown -h now
