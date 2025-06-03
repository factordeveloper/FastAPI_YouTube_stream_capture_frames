from fastapi import FastAPI, HTTPException, BackgroundTasks, Response
from fastapi.responses import FileResponse
from pydantic import BaseModel
import cv2
import numpy as np
import pymongo
from pymongo import MongoClient
from datetime import datetime
import asyncio
import threading
import base64
import yt_dlp
import logging
import io
import os
import time
from PIL import Image
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware



# Configuración de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="YouTube Stream Frame Capture", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Permite todos los orígenes
    allow_credentials=True,
    allow_methods=["*"],  # Permite todos los métodos HTTP
    allow_headers=["*"],  # Permite todos los headers
)


# Configuración de MongoDB (solo para metadatos)
MONGO_URL = "mongodb://localhost:27017"
DATABASE_NAME = "youtube_stream_frames"
COLLECTION_NAME = "frames"

# Configuración de carpeta local para imágenes
FRAMES_FOLDER = "captured_frames"
os.makedirs(FRAMES_FOLDER, exist_ok=True)

# Cliente MongoDB
client = MongoClient(MONGO_URL)
db = client[DATABASE_NAME]
collection = db[COLLECTION_NAME]

# Variable global para controlar la captura
capture_active = False
capture_thread = None

class StreamRequest(BaseModel):
    youtube_url: str
    duration_minutes: int = 60  # Duración en minutos (0 = infinito)

class FrameResponse(BaseModel):
    message: str
    total_frames: int = 0

def get_stream_url(youtube_url: str) -> str:
    """Obtiene la URL del stream en vivo de YouTube"""
    ydl_opts = {
        'format': 'best[ext=mp4]/best',
        'quiet': True,
        'no_warnings': True,
    }
    
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        try:
            info = ydl.extract_info(youtube_url, download=False)
            if info.get('is_live'):
                return info['url']
            else:
                # Si no es un stream en vivo, usa la mejor calidad disponible
                return info['url']
        except Exception as e:
            logger.error(f"Error extracting stream URL: {e}")
            raise HTTPException(status_code=400, detail=f"Error al obtener URL del stream: {str(e)}")

def save_frame_to_local(frame: np.ndarray, timestamp: datetime) -> dict:
    """Guarda un fotograma en carpeta local y metadatos en MongoDB"""
    try:
        # Crear nombre de archivo con timestamp
        filename = f"frame_{timestamp.strftime('%Y%m%d_%H%M%S_%f')}.jpg"
        filepath = os.path.join(FRAMES_FOLDER, filename)
        
        # Convertir frame de OpenCV a formato RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        
        # Guardar imagen en carpeta local
        pil_image.save(filepath, format='JPEG', quality=85)
        
        # Obtener tamaño del archivo
        file_size = os.path.getsize(filepath)
        
        # Guardar metadatos en MongoDB
        frame_doc = {
            "filename": filename,
            "filepath": filepath,
            "timestamp": timestamp,
            "size": file_size
        }
        
        result = collection.insert_one(frame_doc)
        frame_doc["_id"] = str(result.inserted_id)
        
        logger.info(f"Frame guardado: {filepath}")
        return frame_doc
        
    except Exception as e:
        logger.error(f"Error guardando frame: {e}")
        return None

def capture_frames(stream_url: str, duration_minutes: int):
    """Función que captura fotogramas del stream"""
    global capture_active
    
    try:
        # Abrir el stream
        cap = cv2.VideoCapture(stream_url)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reducir buffer para menor latencia
        
        if not cap.isOpened():
            logger.error("No se pudo abrir el stream")
            return
        
        logger.info("Iniciando captura de fotogramas...")
        start_time = datetime.now()
        frame_count = 0
        
        while capture_active:
            ret, frame = cap.read()
            if not ret:
                logger.warning("No se pudo leer frame del stream")
                # Intentar reconectar
                cap.release()
                cap = cv2.VideoCapture(stream_url)
                continue
            
            # Capturar timestamp exacto
            current_time = datetime.now()
            
            # Guardar frame localmente
            frame_doc = save_frame_to_local(frame, current_time)
            if frame_doc:
                frame_count += 1
                logger.info(f"Frame {frame_count} capturado a las {current_time}")
            
            # Verificar duración si está especificada
            if duration_minutes > 0:
                elapsed = (current_time - start_time).total_seconds() / 60
                if elapsed >= duration_minutes:
                    logger.info(f"Duración completada: {duration_minutes} minutos")
                    break
            
            # Esperar 1 segundo para el próximo frame
            time.sleep(1)
        
        cap.release()
        logger.info(f"Captura finalizada. Total de frames: {frame_count}")
        
    except Exception as e:
        logger.error(f"Error en captura de frames: {e}")
    finally:
        capture_active = False

@app.post("/start-capture", response_model=FrameResponse)
async def start_capture(request: StreamRequest, background_tasks: BackgroundTasks):
    """Inicia la captura de fotogramas del stream de YouTube"""
    global capture_active, capture_thread
    
    if capture_active:
        raise HTTPException(status_code=400, detail="Ya hay una captura en progreso")
    
    try:
        # Obtener URL del stream
        stream_url = get_stream_url(request.youtube_url)
        logger.info(f"Stream URL obtenida: {stream_url[:100]}...")
        
        # Iniciar captura en hilo separado
        capture_active = True
        capture_thread = threading.Thread(
            target=capture_frames,
            args=(stream_url, request.duration_minutes),
            daemon=True
        )
        capture_thread.start()
        
        return FrameResponse(
            message=f"Captura iniciada. Duración: {'infinita' if request.duration_minutes == 0 else f'{request.duration_minutes} minutos'}. Imágenes guardándose en: {os.path.abspath(FRAMES_FOLDER)}"
        )
        
    except Exception as e:
        capture_active = False
        logger.error(f"Error iniciando captura: {e}")
        raise HTTPException(status_code=500, detail=f"Error al iniciar captura: {str(e)}")

@app.post("/stop-capture", response_model=FrameResponse)
async def stop_capture():
    """Detiene la captura de fotogramas"""
    global capture_active
    
    if not capture_active:
        raise HTTPException(status_code=400, detail="No hay captura en progreso")
    
    capture_active = False
    
    # Contar frames guardados
    total_frames = collection.count_documents({})
    
    return FrameResponse(
        message="Captura detenida",
        total_frames=total_frames
    )

@app.get("/status")
async def get_status():
    """Obtiene el estado actual de la captura"""
    total_frames = collection.count_documents({})
    latest_frame = collection.find_one(sort=[("timestamp", -1)])
    
    return {
        "capture_active": capture_active,
        "total_frames": total_frames,
        "latest_frame_time": latest_frame["timestamp"] if latest_frame else None,
        "frames_folder": os.path.abspath(FRAMES_FOLDER),
        "database_status": "connected" if client.admin.command('ping') else "disconnected"
    }

@app.get("/frames")
async def get_frames(limit: int = 10, skip: int = 0):
    """Obtiene lista de fotogramas guardados"""
    frames = list(collection.find(
        {},
        {"_id": 0, "filename": 1, "filepath": 1, "timestamp": 1, "size": 1}
    ).sort("timestamp", -1).skip(skip).limit(limit))
    
    return {
        "frames": frames,
        "total": collection.count_documents({}),
        "frames_folder": os.path.abspath(FRAMES_FOLDER)
    }

@app.get("/frame/{filename}")
async def get_frame_image(filename: str):
    """Obtiene una imagen específica por su nombre de archivo"""
    try:
        filepath = os.path.join(FRAMES_FOLDER, filename)
        
        if not os.path.exists(filepath):
            raise HTTPException(status_code=404, detail="Frame no encontrado")
        
        return FileResponse(
            path=filepath,
            media_type="image/jpeg",
            filename=filename
        )
    except Exception as e:
        raise HTTPException(status_code=404, detail="Frame no encontrado")

@app.delete("/frames")
async def delete_all_frames():
    """Elimina todos los fotogramas guardados (archivos locales y metadatos)"""
    try:
        deleted_files = 0
        
        # Eliminar archivos locales
        for frame in collection.find():
            filepath = frame.get("filepath")
            if filepath and os.path.exists(filepath):
                os.remove(filepath)
                deleted_files += 1
        
        # Eliminar documentos de la colección
        result = collection.delete_many({})
        
        return {
            "message": f"Eliminados {deleted_files} archivos y {result.deleted_count} registros",
            "deleted_files": deleted_files,
            "deleted_records": result.deleted_count
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error eliminando frames: {str(e)}")

@app.get("/frames-folder")
async def get_frames_folder_info():
    """Obtiene información sobre la carpeta de frames"""
    try:
        folder_path = os.path.abspath(FRAMES_FOLDER)
        files = []
        total_size = 0
        
        if os.path.exists(FRAMES_FOLDER):
            for filename in os.listdir(FRAMES_FOLDER):
                if filename.endswith('.jpg'):
                    filepath = os.path.join(FRAMES_FOLDER, filename)
                    file_size = os.path.getsize(filepath)
                    file_time = datetime.fromtimestamp(os.path.getmtime(filepath))
                    
                    files.append({
                        "filename": filename,
                        "size": file_size,
                        "modified_time": file_time
                    })
                    total_size += file_size
        
        return {
            "folder_path": folder_path,
            "total_files": len(files),
            "total_size_bytes": total_size,
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "files": sorted(files, key=lambda x: x["modified_time"], reverse=True)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error obteniendo info de carpeta: {str(e)}")

@app.on_event("startup")
async def startup_event():
    """Configuración inicial al iniciar la aplicación"""
    try:
        # Verificar conexión a MongoDB
        client.admin.command('ping')
        logger.info("Conectado a MongoDB exitosamente")
        
        # Crear carpeta de frames si no existe
        os.makedirs(FRAMES_FOLDER, exist_ok=True)
        logger.info(f"Carpeta de frames: {os.path.abspath(FRAMES_FOLDER)}")
        
        # Crear índices
        collection.create_index("timestamp")
        collection.create_index("filename")
        
    except Exception as e:
        logger.error(f"Error conectando a MongoDB: {e}")

@app.on_event("shutdown")
async def shutdown_event():
    """Limpieza al cerrar la aplicación"""
    global capture_active
    capture_active = False
    client.close()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)