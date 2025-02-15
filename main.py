from fastapi import FastAPI, Query, HTTPException
import subprocess
import os
import requests
import json
from pydantic import BaseModel
from typing import Optional
import openai
from datetime import datetime
import sqlite3
import whisper
import markdown
import pandas as pd
from PIL import Image
from bs4 import BeautifulSoup

def validate_path(path: str):
    if not path.startswith("/data/"):
        raise HTTPException(status_code=403, detail="Access outside /data/ is prohibited")

def get_most_recent_log_files(directory, count=10):
    log_files = [f for f in os.listdir(directory) if f.endswith(".log")]
    log_files.sort(key=lambda f: os.path.getmtime(os.path.join(directory, f)), reverse=True)
    return log_files[:count]

def extract_first_lines(log_files, directory):
    first_lines = []
    for log_file in log_files:
        with open(os.path.join(directory, log_file), "r") as f:
            first_line = f.readline().strip()
            first_lines.append(first_line)
    return first_lines

app = FastAPI()

openai.api_key = os.environ.get("AIPROXY_TOKEN")

class TaskRequest(BaseModel):
    task: str

@app.post("/run")
def run_task(task: str = Query(..., description="Task description in plain English")):
    try:
        if "fetch" in task.lower() and "api" in task.lower():
            url = task.split("from")[-1].strip()
            output_file = "/data/api-output.json"
            response = requests.get(url)
            if response.status_code != 200:
                raise HTTPException(status_code=500, detail="Failed to fetch API data")
            with open(output_file, "w") as f:
                json.dump(response.json(), f, indent=4)
            return {"status": "success", "message": f"Data saved to {output_file}"}
        
        if "run sql" in task.lower() and "database" in task.lower():
            query = task.split("run sql")[-1].strip()
            db_path = "/data/database.db"
            validate_path(db_path)
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute(query)
            result = cursor.fetchall()
            conn.close()
            return {"status": "success", "result": result}
        
        if "scrape" in task.lower():
            url = task.split("scrape")[-1].strip()
            response = requests.get(url)
            soup = BeautifulSoup(response.text, "html.parser")
            title = soup.title.string if soup.title else "No Title"
            return {"status": "success", "title": title}
        
        if "resize image" in task.lower():
            image_path = "/data/image.png"
            validate_path(image_path)
            img = Image.open(image_path)
            img = img.resize((img.width // 2, img.height // 2))
            img.save("/data/image-resized.png")
            return {"status": "success", "message": "Image resized successfully"}
        
        if "transcribe" in task.lower():
            model = whisper.load_model("base")
            result = model.transcribe("/data/audio.mp3")
            with open("/data/audio-transcription.txt", "w") as f:
                f.write(result["text"])
            return {"status": "success", "message": "Audio transcribed"}
        
        if "convert markdown" in task.lower():
            md_file = "/data/docs/input.md"
            validate_path(md_file)
            with open(md_file, "r") as f:
                md_content = f.read()
            html_content = markdown.markdown(md_content)
            with open("/data/docs/output.html", "w") as f:
                f.write(html_content)
            return {"status": "success", "message": "Markdown converted to HTML"}
        
        if "filter csv" in task.lower():
            csv_path = "/data/data.csv"
            validate_path(csv_path)
            df = pd.read_csv(csv_path)
            filtered_df = df[df["price"] > 100]
            return {"status": "success", "data": filtered_df.to_dict(orient="records")}
        
        return {"status": "error", "message": "Task not recognized"}
    except openai.error.OpenAIError as e:
        raise HTTPException(status_code=500, detail=f"LLM error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent error: {str(e)}")

@app.get("/read")
def read_file(path: str = Query(..., description="File path to read")):
    validate_path(path)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="File not found")
    try:
        with open(path, "r") as f:
            return {"status": "success", "content": f.read()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"File read error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
