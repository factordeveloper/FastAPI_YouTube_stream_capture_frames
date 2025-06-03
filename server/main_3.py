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

# Configuración de carpetas locales para imágenes
FRAMES_FOLDER = "captured_frames"
TRAINS_FOLDER = "frames_with_trains"
NO_TRAINS_FOLDER = "frames_without_trains"

# Crear carpetas si no existen
for folder in [FRAMES_FOLDER, TRAINS_FOLDER, NO_TRAINS_FOLDER]:
    os.makedirs(folder, exist_ok=True)

# Cliente MongoDB
client = MongoClient(MONGO_URL)
db = client[DATABASE_NAME]
collection = db[COLLECTION_NAME]

# Variable global para controlar la captura
capture_active = False
capture_thread = None

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
    no_trains_detected: int = 0

class DetectionStats(BaseModel):
    total_frames: int
    frames_with_trains: int
    frames_without_trains: int
    detection_accuracy: float

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

def save_frame_to_local(frame: np.ndarray, timestamp: datetime, has_trains: bool, detections: list, annotated_frame: np.ndarray = None) -> dict:
    """Guarda un fotograma en la carpeta correspondiente según si tiene trenes o no"""
    try:
        # Determinar carpeta destino
        target_folder = TRAINS_FOLDER if has_trains else NO_TRAINS_FOLDER
        category = "with_trains" if has_trains else "without_trains"
        
        # Crear nombre de archivo con timestamp
        filename = f"frame_{category}_{timestamp.strftime('%Y%m%d_%H%M%S_%f')}.jpg"
        filepath = os.path.join(target_folder, filename)
        
        # Decidir qué frame guardar (original o anotado)
        frame_to_save = annotated_frame if annotated_frame is not None and has_trains else frame
        
        # Convertir frame de OpenCV a formato RGB
        frame_rgb = cv2.cvtColor(frame_to_save, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        
        # Guardar imagen en carpeta correspondiente
        pil_image.save(filepath, format='JPEG', quality=85)
        
        # Obtener tamaño del archivo
        file_size = os.path.getsize(filepath)
        
        # Guardar metadatos en MongoDB
        frame_doc = {
            "filename": filename,
            "filepath": filepath,
            "timestamp": timestamp,
            "size": file_size,
            "has_trains": has_trains,
            "category": category,
            "detections": detections,
            "detection_count": len(detections)
        }
        
        result = collection.insert_one(frame_doc)
        frame_doc["_id"] = str(result.inserted_id)
        
        logger.info(f"Frame guardado en {category}: {filepath} - Trenes detectados: {len(detections)}")
        return frame_doc
        
    except Exception as e:
        logger.error(f"Error guardando frame: {e}")
        return None

def capture_frames(stream_url: str, duration_minutes: int):
    """Función que captura fotogramas del stream y los clasifica"""
    global capture_active
    
    try:
        # Abrir el stream
        cap = cv2.VideoCapture(stream_url)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reducir buffer para menor latencia
        
        if not cap.isOpened():
            logger.error("No se pudo abrir el stream")
            return
        
        logger.info("Iniciando captura de fotogramas con detección de trenes...")
        start_time = datetime.now()
        frame_count = 0
        trains_detected = 0
        
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
            
            # Detectar trenes en el frame
            has_trains, detections, annotated_frame = detect_trains(frame)
            
            if has_trains:
                trains_detected += 1
            
            # Guardar frame en la carpeta correspondiente
            frame_doc = save_frame_to_local(frame, current_time, has_trains, detections, annotated_frame)
            if frame_doc:
                frame_count += 1
                status = f"con {len(detections)} tren(es)" if has_trains else "sin trenes"
                logger.info(f"Frame {frame_count} capturado a las {current_time} - {status}")
            
            # Verificar duración si está especificada
            if duration_minutes > 0:
                elapsed = (current_time - start_time).total_seconds() / 60
                if elapsed >= duration_minutes:
                    logger.info(f"Duración completada: {duration_minutes} minutos")
                    break
            
            # Esperar 1 segundo para el próximo frame
            time.sleep(1)
        
        cap.release()
        logger.info(f"Captura finalizada. Total de frames: {frame_count}, Frames con trenes: {trains_detected}")
        
    except Exception as e:
        logger.error(f"Error en captura de frames: {e}")
    finally:
        capture_active = False

@app.post("/start-capture", response_model=FrameResponse)
async def start_capture(request: StreamRequest, background_tasks: BackgroundTasks):
    """Inicia la captura de fotogramas del stream de YouTube con detección de trenes"""
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
            message=f"Captura con detección de trenes iniciada. Duración: {'infinita' if request.duration_minutes == 0 else f'{request.duration_minutes} minutos'}. Imágenes con trenes: {os.path.abspath(TRAINS_FOLDER)}, sin trenes: {os.path.abspath(NO_TRAINS_FOLDER)}"
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
    trains_detected = collection.count_documents({"has_trains": True})
    no_trains_detected = collection.count_documents({"has_trains": False})
    
    return FrameResponse(
        message="Captura detenida",
        total_frames=total_frames,
        trains_detected=trains_detected,
        no_trains_detected=no_trains_detected
    )

@app.get("/status")
async def get_status():
    """Obtiene el estado actual de la captura y estadísticas de detección"""
    total_frames = collection.count_documents({})
    trains_detected = collection.count_documents({"has_trains": True})
    no_trains_detected = collection.count_documents({"has_trains": False})
    latest_frame = collection.find_one(sort=[("timestamp", -1)])
    
    return {
        "capture_active": capture_active,
        "total_frames": total_frames,
        "frames_with_trains": trains_detected,
        "frames_without_trains": no_trains_detected,
        "detection_rate": round((trains_detected / total_frames * 100), 2) if total_frames > 0 else 0,
        "latest_frame_time": latest_frame["timestamp"] if latest_frame else None,
        "latest_frame_has_trains": latest_frame.get("has_trains", False) if latest_frame else False,
        "trains_folder": os.path.abspath(TRAINS_FOLDER),
        "no_trains_folder": os.path.abspath(NO_TRAINS_FOLDER),
        "database_status": "connected" if client.admin.command('ping') else "disconnected",
        "yolo_model_loaded": model is not None
    }

@app.get("/frames")
async def get_frames(limit: int = 10, skip: int = 0, filter_trains: bool = None):
    """Obtiene lista de fotogramas guardados con filtro opcional por detección de trenes"""
    query = {}
    if filter_trains is not None:
        query["has_trains"] = filter_trains
    
    frames = list(collection.find(
        query,
        {"_id": 0, "filename": 1, "filepath": 1, "timestamp": 1, "size": 1, 
         "has_trains": 1, "category": 1, "detection_count": 1}
    ).sort("timestamp", -1).skip(skip).limit(limit))
    
    total_frames = collection.count_documents({})
    trains_detected = collection.count_documents({"has_trains": True})
    
    return {
        "frames": frames,
        "total": collection.count_documents(query),
        "total_all_frames": total_frames,
        "total_with_trains": trains_detected,
        "total_without_trains": total_frames - trains_detected,
        "trains_folder": os.path.abspath(TRAINS_FOLDER),
        "no_trains_folder": os.path.abspath(NO_TRAINS_FOLDER)
    }

@app.get("/frame/{filename}")
async def get_frame_image(filename: str):
    """Obtiene una imagen específica por su nombre de archivo"""
    try:
        # Buscar en ambas carpetas
        filepath_trains = os.path.join(TRAINS_FOLDER, filename)
        filepath_no_trains = os.path.join(NO_TRAINS_FOLDER, filename)
        
        filepath = None
        if os.path.exists(filepath_trains):
            filepath = filepath_trains
        elif os.path.exists(filepath_no_trains):
            filepath = filepath_no_trains
        
        if not filepath:
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
    total_frames = collection.count_documents({})
    frames_with_trains = collection.count_documents({"has_trains": True})
    frames_without_trains = collection.count_documents({"has_trains": False})
    
    # Estadísticas adicionales
    avg_detections = list(collection.aggregate([
        {"$match": {"has_trains": True}},
        {"$group": {"_id": None, "avg_detections": {"$avg": "$detection_count"}}}
    ]))
    
    recent_detections = list(collection.find(
        {"has_trains": True},
        {"timestamp": 1, "detection_count": 1, "filename": 1}
    ).sort("timestamp", -1).limit(5))
    
    return DetectionStats(
        total_frames=total_frames,
        frames_with_trains=frames_with_trains,
        frames_without_trains=frames_without_trains,
        detection_accuracy=round((frames_with_trains / total_frames * 100), 2) if total_frames > 0 else 0
    ), {
        "average_trains_per_detection": round(avg_detections[0]["avg_detections"], 2) if avg_detections else 0,
        "recent_train_detections": recent_detections,
        "model_info": {
            "model_loaded": model is not None,
            "model_type": "YOLOv8n" if model else None,
            "train_class_id": TRAIN_CLASSES
        }
    }

@app.delete("/frames")
async def delete_all_frames():
    """Elimina todos los fotogramas guardados (archivos locales y metadatos)"""
    try:
        deleted_files = 0
        
        # Eliminar archivos de ambas carpetas
        for folder in [TRAINS_FOLDER, NO_TRAINS_FOLDER]:
            if os.path.exists(folder):
                for filename in os.listdir(folder):
                    if filename.endswith('.jpg'):
                        filepath = os.path.join(folder, filename)
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

@app.get("/folders-info")
async def get_folders_info():
    """Obtiene información sobre ambas carpetas de frames"""
    try:
        folders_info = {}
        
        for folder_name, folder_path in [
            ("trains", TRAINS_FOLDER),
            ("no_trains", NO_TRAINS_FOLDER)
        ]:
            files = []
            total_size = 0
            
            if os.path.exists(folder_path):
                for filename in os.listdir(folder_path):
                    if filename.endswith('.jpg'):
                        filepath = os.path.join(folder_path, filename)
                        file_size = os.path.getsize(filepath)
                        file_time = datetime.fromtimestamp(os.path.getmtime(filepath))
                        
                        files.append({
                            "filename": filename,
                            "size": file_size,
                            "modified_time": file_time
                        })
                        total_size += file_size
            
            folders_info[folder_name] = {
                "folder_path": os.path.abspath(folder_path),
                "total_files": len(files),
                "total_size_bytes": total_size,
                "total_size_mb": round(total_size / (1024 * 1024), 2),
                "files": sorted(files, key=lambda x: x["modified_time"], reverse=True)[:10]  # Solo los 10 más recientes
            }
        
        return folders_info
        
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
        for folder in [FRAMES_FOLDER, TRAINS_FOLDER, NO_TRAINS_FOLDER]:
            os.makedirs(folder, exist_ok=True)
        
        logger.info(f"Carpeta para frames con trenes: {os.path.abspath(TRAINS_FOLDER)}")
        logger.info(f"Carpeta para frames sin trenes: {os.path.abspath(NO_TRAINS_FOLDER)}")
        
        # Crear índices
        collection.create_index("timestamp")
        collection.create_index("filename")
        collection.create_index("has_trains")
        collection.create_index("category")
        
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
    uvicorn.run(app, host="127.0.0.1", port=8000)