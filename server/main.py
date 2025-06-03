from fastapi import FastAPI, HTTPException, BackgroundTasks, Response  
from pydantic import BaseModel
import cv2
import numpy as np
import pymongo
from pymongo import MongoClient
import gridfs
from datetime import datetime
import asyncio
import threading
import base64
import yt_dlp
import logging
import io
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

# Configuración de MongoDB
MONGO_URL = "mongodb://localhost:27017"
DATABASE_NAME = "youtube_frames"
COLLECTION_NAME = "frames"

# Cliente MongoDB
client = MongoClient(MONGO_URL)
db = client[DATABASE_NAME]
collection = db[COLLECTION_NAME]
fs = gridfs.GridFS(db)

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

def save_frame_to_mongo(frame: np.ndarray, timestamp: datetime) -> str:
    """Guarda un fotograma en MongoDB usando GridFS"""
    try:
        # Convertir frame de OpenCV a imagen PIL
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        
        # Convertir a bytes
        img_buffer = io.BytesIO()
        pil_image.save(img_buffer, format='JPEG', quality=85)
        img_bytes = img_buffer.getvalue()
        
        # Guardar en GridFS
        file_id = fs.put(
            img_bytes,
            filename=f"frame_{timestamp.strftime('%Y%m%d_%H%M%S_%f')}.jpg",
            content_type="image/jpeg",
            timestamp=timestamp
        )
        
        # Guardar metadatos en la colección
        frame_doc = {
            "file_id": file_id,
            "timestamp": timestamp,
            "filename": f"frame_{timestamp.strftime('%Y%m%d_%H%M%S_%f')}.jpg",
            "size": len(img_bytes)
        }
        
        collection.insert_one(frame_doc)
        logger.info(f"Frame guardado con ID: {file_id}")
        return str(file_id)
        
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
            
            # Guardar frame en MongoDB
            file_id = save_frame_to_mongo(frame, current_time)
            if file_id:
                frame_count += 1
                logger.info(f"Frame {frame_count} capturado a las {current_time}")
            
            # Verificar duración si está especificada
            if duration_minutes > 0:
                elapsed = (current_time - start_time).total_seconds() / 60
                if elapsed >= duration_minutes:
                    logger.info(f"Duración completada: {duration_minutes} minutos")
                    break
            
            # Esperar 1 segundo para el próximo frame
            asyncio.sleep(1)
        
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
            message=f"Captura iniciada. Duración: {'infinita' if request.duration_minutes == 0 else f'{request.duration_minutes} minutos'}"
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
        "database_status": "connected" if client.admin.command('ping') else "disconnected"
    }

@app.get("/frames")
async def get_frames(limit: int = 10, skip: int = 0):
    """Obtiene lista de fotogramas guardados"""
    frames = list(collection.find(
        {},
        {"_id": 0, "file_id": 1, "timestamp": 1, "filename": 1, "size": 1}
    ).sort("timestamp", -1).skip(skip).limit(limit))
    
    return {
        "frames": frames,
        "total": collection.count_documents({})
    }

@app.get("/frame/{file_id}")
async def get_frame_image(file_id: str):
    """Obtiene una imagen específica por su ID"""
    try:
        from bson import ObjectId
        grid_out = fs.get(ObjectId(file_id))
        
        return Response(
            content=grid_out.read(),
            media_type="image/jpeg",
            headers={"Content-Disposition": f"inline; filename={grid_out.filename}"}
        )
    except Exception as e:
        raise HTTPException(status_code=404, detail="Frame no encontrado")

@app.delete("/frames")
async def delete_all_frames():
    """Elimina todos los fotogramas guardados"""
    try:
        # Eliminar archivos de GridFS
        for frame in collection.find():
            fs.delete(frame["file_id"])
        
        # Eliminar documentos de la colección
        result = collection.delete_many({})
        
        return {
            "message": f"Eliminados {result.deleted_count} fotogramas",
            "deleted_count": result.deleted_count
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error eliminando frames: {str(e)}")

@app.on_event("startup")
async def startup_event():
    """Configuración inicial al iniciar la aplicación"""
    try:
        # Verificar conexión a MongoDB
        client.admin.command('ping')
        logger.info("Conectado a MongoDB exitosamente")
        
        # Crear índices
        collection.create_index("timestamp")
        collection.create_index("file_id")
        
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
    uvicorn.run(app, host="127.0.0.1", port=8000)