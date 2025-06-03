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
from ultralytics import YOLO
import torch


# Configuración de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="YouTube Stream Frame Capture with Train Detection", version="1.0.0")

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

# Configuración de carpetas locales para imágenes - SOLO TRENES
FRAMES_FOLDER = "captured_frames"
TRAINS_FOLDER = "frames_with_trains"
# Eliminamos NO_TRAINS_FOLDER ya que no guardaremos frames sin trenes

# Crear carpetas si no existen - SOLO PARA TRENES
for folder in [FRAMES_FOLDER, TRAINS_FOLDER]:
    os.makedirs(folder, exist_ok=True)

# Cliente MongoDB
client = MongoClient(MONGO_URL)
db = client[DATABASE_NAME]
collection = db[COLLECTION_NAME]

# Variable global para controlar la captura
capture_active = False
capture_thread = None

# Contadores globales
frames_processed = 0
trains_detected = 0
frames_discarded = 0

# Inicializar modelo YOLO
try:
    # Cargar modelo YOLO preentrenado (puedes usar yolov8n.pt, yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt)
    model = YOLO('yolov8n.pt')  # Modelo más ligero para mayor velocidad
    logger.info("Modelo YOLO cargado exitosamente")
except Exception as e:
    logger.error(f"Error cargando modelo YOLO: {e}")
    model = None

# Clases de YOLO que consideramos como "trenes"
# En COCO dataset: 6 = 'train'
TRAIN_CLASSES = [6]  # Clase 6 corresponde a 'train' en COCO dataset

class StreamRequest(BaseModel):
    youtube_url: str
    duration_minutes: int = 60  # Duración en minutos (0 = infinito)

class FrameResponse(BaseModel):
    message: str
    total_frames: int = 0
    trains_detected: int = 0
    frames_discarded: int = 0

class DetectionStats(BaseModel):
    total_frames_processed: int
    frames_with_trains_saved: int
    frames_discarded: int
    detection_rate: float
    save_rate: float

def detect_trains(frame: np.ndarray) -> tuple[bool, list, np.ndarray]:
    """
    Detecta trenes en un frame usando YOLO
    Retorna: (has_trains, detections, annotated_frame)
    """
    try:
        if model is None:
            return False, [], frame
        
        # Realizar detección
        results = model(frame, verbose=False)
        
        has_trains = False
        detections = []
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # Obtener clase y confianza
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    
                    # Verificar si es un tren y tiene suficiente confianza
                    if cls in TRAIN_CLASSES and conf > 0.5:  # Umbral de confianza
                        has_trains = True
                        
                        # Obtener coordenadas de la caja
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        
                        detections.append({
                            'class': cls,
                            'confidence': conf,
                            'bbox': [x1, y1, x2, y2],
                            'class_name': 'train'
                        })
        
        # Crear frame anotado si hay detecciones
        annotated_frame = frame.copy()
        if has_trains:
            annotated_frame = results[0].plot()  # YOLO dibuja las cajas automáticamente
        
        return has_trains, detections, annotated_frame
        
    except Exception as e:
        logger.error(f"Error en detección de trenes: {e}")
        return False, [], frame

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

def save_train_frame(frame: np.ndarray, timestamp: datetime, detections: list, annotated_frame: np.ndarray) -> dict:
    """Guarda SOLO los fotogramas que contienen trenes"""
    try:
        # Crear nombre de archivo con timestamp
        filename = f"frame_with_trains_{timestamp.strftime('%Y%m%d_%H%M%S_%f')}.jpg"
        filepath = os.path.join(TRAINS_FOLDER, filename)
        
        # Guardar el frame anotado (con las cajas de detección)
        frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        
        # Guardar imagen
        pil_image.save(filepath, format='JPEG', quality=85)
        
        # Obtener tamaño del archivo
        file_size = os.path.getsize(filepath)
        
        # Guardar metadatos en MongoDB
        frame_doc = {
            "filename": filename,
            "filepath": filepath,
            "timestamp": timestamp,
            "size": file_size,
            "has_trains": True,
            "detections": detections,
            "detection_count": len(detections)
        }
        
        result = collection.insert_one(frame_doc)
        frame_doc["_id"] = str(result.inserted_id)
        
        logger.info(f"Frame con {len(detections)} tren(es) guardado: {filename}")
        return frame_doc
        
    except Exception as e:
        logger.error(f"Error guardando frame con trenes: {e}")
        return None

def capture_frames(stream_url: str, duration_minutes: int):
    """Función que captura fotogramas del stream cada 5 segundos y guarda SOLO los que tienen trenes"""
    global capture_active, frames_processed, trains_detected, frames_discarded
    
    try:
        # Abrir el stream
        cap = cv2.VideoCapture(stream_url)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reducir buffer para menor latencia
        
        if not cap.isOpened():
            logger.error("No se pudo abrir el stream")
            return
        
        logger.info("Iniciando captura de fotogramas cada 5 segundos con detección de trenes...")
        logger.info("SOLO se guardarán los frames que contengan trenes")
        start_time = datetime.now()
        
        # Reiniciar contadores
        frames_processed = 0
        trains_detected = 0
        frames_discarded = 0
        
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
            frames_processed += 1
            
            # Detectar trenes en el frame
            has_trains, detections, annotated_frame = detect_trains(frame)
            
            if has_trains:
                # GUARDAR: Frame contiene trenes
                frame_doc = save_train_frame(frame, current_time, detections, annotated_frame)
                if frame_doc:
                    trains_detected += 1
                    logger.info(f"✅ Frame {frames_processed} - GUARDADO - {len(detections)} tren(es) detectado(s)")
            else:
                # DESCARTAR: Frame sin trenes
                frames_discarded += 1
                logger.info(f"❌ Frame {frames_processed} - DESCARTADO - Sin trenes detectados")
            
            # Verificar duración si está especificada
            if duration_minutes > 0:
                elapsed = (current_time - start_time).total_seconds() / 60
                if elapsed >= duration_minutes:
                    logger.info(f"Duración completada: {duration_minutes} minutos")
                    break
            
            # Esperar 5 segundos para el próximo frame
            time.sleep(5)
        
        cap.release()
        logger.info(f"Captura finalizada.")
        logger.info(f"Frames procesados: {frames_processed}")
        logger.info(f"Frames guardados (con trenes): {trains_detected}")
        logger.info(f"Frames descartados (sin trenes): {frames_discarded}")
        logger.info(f"Tasa de detección: {(trains_detected/frames_processed*100):.1f}%" if frames_processed > 0 else "0%")
        
    except Exception as e:
        logger.error(f"Error en captura de frames: {e}")
    finally:
        capture_active = False

@app.post("/start-capture", response_model=FrameResponse)
async def start_capture(request: StreamRequest, background_tasks: BackgroundTasks):
    """Inicia la captura de fotogramas del stream de YouTube cada 5 segundos (SOLO guarda frames con trenes)"""
    global capture_active, capture_thread
    
    if capture_active:
        raise HTTPException(status_code=400, detail="Ya hay una captura en progreso")
    
    if model is None:
        raise HTTPException(status_code=500, detail="Modelo YOLO no está disponible")
    
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
            message=f"Captura iniciada - Captura cada 5 segundos - SOLO guarda frames con trenes. Duración: {'infinita' if request.duration_minutes == 0 else f'{request.duration_minutes} minutos'}. Carpeta: {os.path.abspath(TRAINS_FOLDER)}"
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
    
    # Contar frames guardados (solo los que tienen trenes)
    total_saved = collection.count_documents({"has_trains": True})
    
    return FrameResponse(
        message="Captura detenida",
        total_frames=frames_processed,
        trains_detected=trains_detected,
        frames_discarded=frames_discarded
    )

@app.get("/status")
async def get_status():
    """Obtiene el estado actual de la captura y estadísticas de detección"""
    total_saved = collection.count_documents({"has_trains": True})
    latest_frame = collection.find_one(sort=[("timestamp", -1)])
    
    return {
        "capture_active": capture_active,
        "frames_processed": frames_processed,
        "frames_saved_with_trains": total_saved,
        "frames_discarded": frames_discarded,
        "detection_rate": round((trains_detected / frames_processed * 100), 2) if frames_processed > 0 else 0,
        "save_rate": round((total_saved / frames_processed * 100), 2) if frames_processed > 0 else 0,
        "latest_frame_time": latest_frame["timestamp"] if latest_frame else None,
        "latest_detection_count": latest_frame.get("detection_count", 0) if latest_frame else 0,
        "trains_folder": os.path.abspath(TRAINS_FOLDER),
        "capture_interval": "5 segundos",
        "storage_policy": "Solo frames con trenes",
        "database_status": "connected" if client.admin.command('ping') else "disconnected",
        "yolo_model_loaded": model is not None
    }

@app.get("/frames")
async def get_frames(limit: int = 10, skip: int = 0):
    """Obtiene lista de fotogramas guardados (solo los que contienen trenes)"""
    frames = list(collection.find(
        {"has_trains": True},
        {"_id": 0, "filename": 1, "filepath": 1, "timestamp": 1, "size": 1, 
         "detection_count": 1, "detections": 1}
    ).sort("timestamp", -1).skip(skip).limit(limit))
    
    total_frames = collection.count_documents({"has_trains": True})
    
    return {
        "frames": frames,
        "total_saved": total_frames,
        "frames_processed": frames_processed,
        "frames_discarded": frames_discarded,
        "trains_folder": os.path.abspath(TRAINS_FOLDER),
        "note": "Solo se muestran frames con trenes detectados"
    }

@app.get("/frame/{filename}")
async def get_frame_image(filename: str):
    """Obtiene una imagen específica por su nombre de archivo"""
    try:
        filepath = os.path.join(TRAINS_FOLDER, filename)
        
        if not os.path.exists(filepath):
            raise HTTPException(status_code=404, detail="Frame no encontrado")
        
        return FileResponse(
            path=filepath,
            media_type="image/jpeg",
            filename=filename
        )
    except Exception as e:
        raise HTTPException(status_code=404, detail="Frame no encontrado")

@app.get("/detection-stats")
async def get_detection_stats():
    """Obtiene estadísticas detalladas de detección"""
    total_saved = collection.count_documents({"has_trains": True})
    
    # Estadísticas adicionales
    avg_detections = list(collection.aggregate([
        {"$match": {"has_trains": True}},
        {"$group": {"_id": None, "avg_detections": {"$avg": "$detection_count"}}}
    ]))
    
    recent_detections = list(collection.find(
        {"has_trains": True},
        {"timestamp": 1, "detection_count": 1, "filename": 1}
    ).sort("timestamp", -1).limit(5))
    
    stats = DetectionStats(
        total_frames_processed=frames_processed,
        frames_with_trains_saved=total_saved,
        frames_discarded=frames_discarded,
        detection_rate=round((trains_detected / frames_processed * 100), 2) if frames_processed > 0 else 0,
        save_rate=round((total_saved / frames_processed * 100), 2) if frames_processed > 0 else 0
    )
    
    additional_info = {
        "average_trains_per_detection": round(avg_detections[0]["avg_detections"], 2) if avg_detections else 0,
        "recent_train_detections": recent_detections,
        "model_info": {
            "model_loaded": model is not None,
            "model_type": "YOLOv8n" if model else None,
            "train_class_id": TRAIN_CLASSES,
            "confidence_threshold": 0.5
        },
        "capture_settings": {
            "interval_seconds": 5,
            "storage_policy": "Solo frames con trenes",
            "auto_discard": True
        }
    }
    
    return {"stats": stats, "additional_info": additional_info}

@app.delete("/frames")
async def delete_all_frames():
    """Elimina todos los fotogramas guardados (archivos locales y metadatos)"""
    try:
        deleted_files = 0
        
        # Eliminar archivos de la carpeta de trenes
        if os.path.exists(TRAINS_FOLDER):
            for filename in os.listdir(TRAINS_FOLDER):
                if filename.endswith('.jpg'):
                    filepath = os.path.join(TRAINS_FOLDER, filename)
                    os.remove(filepath)
                    deleted_files += 1
        
        # Eliminar documentos de la colección
        result = collection.delete_many({})
        
        # Reiniciar contadores
        global frames_processed, trains_detected, frames_discarded
        frames_processed = 0
        trains_detected = 0
        frames_discarded = 0
        
        return {
            "message": f"Eliminados {deleted_files} archivos y {result.deleted_count} registros",
            "deleted_files": deleted_files,
            "deleted_records": result.deleted_count,
            "note": "Contadores reiniciados"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error eliminando frames: {str(e)}")

@app.get("/folders-info")
async def get_folders_info():
    """Obtiene información sobre la carpeta de frames con trenes"""
    try:
        files = []
        total_size = 0
        
        if os.path.exists(TRAINS_FOLDER):
            for filename in os.listdir(TRAINS_FOLDER):
                if filename.endswith('.jpg'):
                    filepath = os.path.join(TRAINS_FOLDER, filename)
                    file_size = os.path.getsize(filepath)
                    file_time = datetime.fromtimestamp(os.path.getmtime(filepath))
                    
                    files.append({
                        "filename": filename,
                        "size": file_size,
                        "modified_time": file_time
                    })
                    total_size += file_size
        
        return {
            "trains_folder": {
                "folder_path": os.path.abspath(TRAINS_FOLDER),
                "total_files": len(files),
                "total_size_bytes": total_size,
                "total_size_mb": round(total_size / (1024 * 1024), 2),
                "files": sorted(files, key=lambda x: x["modified_time"], reverse=True)[:10]  # Solo los 10 más recientes
            },
            "capture_info": {
                "frames_processed": frames_processed,
                "frames_saved": trains_detected,
                "frames_discarded": frames_discarded,
                "storage_policy": "Solo frames con trenes detectados",
                "capture_interval": "5 segundos"
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error obteniendo info de carpetas: {str(e)}")

@app.on_event("startup")
async def startup_event():
    """Configuración inicial al iniciar la aplicación"""
    try:
        # Verificar conexión a MongoDB
        client.admin.command('ping')
        logger.info("Conectado a MongoDB exitosamente")
        
        # Crear carpetas si no existen
        for folder in [FRAMES_FOLDER, TRAINS_FOLDER]:
            os.makedirs(folder, exist_ok=True)
        
        logger.info(f"Carpeta para frames con trenes: {os.path.abspath(TRAINS_FOLDER)}")
        logger.info("⚠️  CONFIGURACIÓN: Solo se guardarán frames CON trenes detectados")
        logger.info("⚠️  CONFIGURACIÓN: Captura cada 5 segundos")
        logger.info("⚠️  CONFIGURACIÓN: Frames sin trenes se descartan automáticamente")
        
        # Crear índices
        collection.create_index("timestamp")
        collection.create_index("filename")
        collection.create_index("has_trains")
        
        # Verificar modelo YOLO
        if model is not None:
            logger.info("Modelo YOLO listo para detección de trenes")
        else:
            logger.warning("Modelo YOLO no disponible - funcionará sin detección")
        
    except Exception as e:
        logger.error(f"Error en configuración inicial: {e}")

@app.on_event("shutdown")
async def shutdown_event():
    """Limpieza al cerrar la aplicación"""
    global capture_active
    capture_active = False
    client.close()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)