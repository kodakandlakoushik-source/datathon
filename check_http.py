import urllib.request

try:
    health_resp = urllib.request.urlopen("http://localhost:8501/_stcore/health")
    print(f"Health check status: {health_resp.status} - {health_resp.read().decode().strip()}")
    
    page_resp = urllib.request.urlopen("http://localhost:8501")
    body = page_resp.read().decode("utf-8")
    print(f"Main page status: {page_resp.status} - Length: {len(body)} bytes")
    print("Page contains 'CivicPulse AI':", "CivicPulse AI" in body)
except Exception as e:
    print(f"HTTP verification failed: {e}")
