# Documentación del Proyecto

Este documento proporciona una guía completa sobre la estructura, tecnologías utilizadas, configuración de variables de entorno, y despliegue de la aplicación. Está diseñado para ayudar a desarrolladores a entender, configurar y ejecutar el proyecto de manera eficiente.

## 1. Estructura del Proyecto

El proyecto se compone de dos servicios principales: un backend desarrollado con FastAPI y un frontend desarrollado con Next.js. Ambos servicios están contenidos en sus respectivos directorios (`backend` y `frontend`) dentro del directorio raíz del proyecto. Además, se incluyen archivos de configuración para Docker y Render.

```
.new-tfg-main-main/
├── backend/                  # Contiene el código fuente del backend (FastAPI)
│   ├── app/                  # Lógica principal de la aplicación
│   │   ├── api/              # Endpoints de la API
│   │   ├── core/             # Configuración, autenticación, base de datos
│   │   ├── crud/             # Operaciones CRUD para la base de datos
│   │   ├── main.py           # Punto de entrada de la aplicación FastAPI
│   │   ├── models/           # Modelos de la base de datos (SQLAlchemy)
│   │   ├── schemas/          # Esquemas de datos (Pydantic)
│   │   └── utils/            # Utilidades y funciones auxiliares
│   ├── Dockerfile            # Dockerfile para construir la imagen del backend
│   └── requirements.txt      # Dependencias de Python
├── frontend/                 # Contiene el código fuente del frontend (Next.js)
│   ├── app/                  # Páginas y rutas de la aplicación Next.js
│   ├── components/           # Componentes reutilizables de UI
│   ├── contexts/             # Contextos de React
│   ├── hooks/                # Hooks personalizados de React
│   ├── lib/                  # Librerías y utilidades (ej. configuración de API)
│   ├── public/               # Archivos estáticos
│   ├── styles/               # Estilos globales
│   ├── Dockerfile            # Dockerfile para construir la imagen del frontend
│   ├── package.json          # Dependencias de Node.js y scripts
│   ├── package-lock.json     # Bloqueo de dependencias
│   └── ...                   # Otros archivos de configuración de Next.js/TypeScript
├── docker-compose.yml        # Configuración de Docker Compose para orquestar servicios
└── render.yaml               # Configuración de despliegue para Render.com
```

## 2. Tecnologías Utilizadas

### 2.1. Backend

El backend está construido con **FastAPI**, un framework web moderno y rápido para construir APIs con Python 3.7+. Utiliza **Pydantic** para la validación de datos y la serialización, y **SQLAlchemy** como ORM para interactuar con la base de datos. La base de datos utilizada es **PostgreSQL**.

### 2.2. Frontend

El frontend está desarrollado con **Next.js**, un framework de React para la construcción de aplicaciones web con renderizado del lado del servidor (SSR) y generación de sitios estáticos (SSG). Utiliza **TypeScript** para un desarrollo más robusto y escalable. La gestión de paquetes se realiza con **npm**.

## 3. Configuración de Variables de Entorno

Las variables de entorno son cruciales para la configuración de la aplicación, especialmente para la conexión a servicios externos y la gestión de secretos. Se utilizan archivos `.env` para el backend y `.env.local` para el frontend.

### 3.1. Archivo `.env` (Backend)

El archivo `.env` en el directorio `backend` contiene las variables de entorno necesarias para el correcto funcionamiento del servicio FastAPI. Estas variables se cargan a través de la clase `Settings` definida en `backend/app/core/config.py`.

Las siguientes variables deben ser definidas en el archivo `.env` del backend:

*   `CLIENT_ID`: ID de cliente para la autenticación con servicios externos (ej. Google OAuth).
*   `CLIENT_SECRET`: Secreto de cliente para la autenticación con servicios externos.
*   `REDIRECT_URI`: URI de redirección configurada en el proveedor de autenticación (ej. Google Cloud Console).
*   `APP_SECRET_KEY`: Clave secreta utilizada por la aplicación para la seguridad de sesiones y otros propósitos criptográficos.
*   `DRIVE_FOLDER_NAME`: Nombre de la carpeta en Google Drive donde se almacenarán los notebooks generados.
*   `COLAB_URL`: URL base para Google Colab (ej. `https://colab.research.google.com/drive/XYZ`).
*   `NOTEBOOK_ID`: ID de un notebook de plantilla en Google Colab.
*   `DB_URL`: URL de conexión a la base de datos PostgreSQL. En entornos de producción como Render, esta variable puede ser inyectada automáticamente.
*   `HASH_SECRET_KEY`: Clave secreta utilizada para el hashing de contraseñas u otros datos sensibles.
*   `FRONTEND_URL`: URL del frontend de la aplicación, utilizada para redirecciones o CORS.

Ejemplo de `.env` para el backend:

```env
CLIENT_ID=tu_client_id_aqui
CLIENT_SECRET=tu_client_secret_aqui
REDIRECT_URI=http://localhost:8000/auth/callback
APP_SECRET_KEY=una_clave_secreta_muy_segura
DRIVE_FOLDER_NAME=ColaboAutomation
COLAB_URL=https://colab.research.google.com/drive/tu_id_de_colab
NOTEBOOK_ID=tu_id_de_notebook_plantilla
DB_URL=postgresql://user:password@host:port/database
HASH_SECRET_KEY=otra_clave_secreta_para_hashing
FRONTEND_URL=http://localhost:3000
```

### 3.2. Archivo `.env.local` (Frontend)

El archivo `.env.local` en el directorio `frontend` es utilizado por Next.js para cargar variables de entorno que son accesibles en el lado del cliente (prefijo `NEXT_PUBLIC_`). En este proyecto, se utiliza para definir la URL base del backend.

*   `NEXT_PUBLIC_BACKEND_URL`: La URL base del servicio backend. Esta es la dirección a la que el frontend enviará las solicitudes de API.

Ejemplo de `.env.local` para el frontend:

```env
NEXT_PUBLIC_BACKEND_URL=http://localhost:8000
```

## 4. Dockerización y Despliegue

El proyecto está configurado para ser ejecutado y desplegado utilizando Docker y Docker Compose, y para despliegues en la nube, se proporciona un archivo `render.yaml` para Render.com.

### 4.1. Dockerfiles

#### 4.1.1. `backend/Dockerfile`

Este Dockerfile construye la imagen Docker para el servicio backend. Se basa en una imagen oficial de Python 3.10-slim para reducir el tamaño de la imagen final. Los pasos principales son:

1.  Establecer el directorio de trabajo a `/app`.
2.  Copiar `requirements.txt` e instalar las dependencias de Python.
3.  Copiar el resto del código de la aplicación.
4.  Definir el comando de inicio para ejecutar la aplicación FastAPI con Uvicorn en el puerto 8000.

```dockerfile
# backend/Dockerfile
FROM python:3.10-slim

WORKDIR /app

# Copiar primero las dependencias
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Luego copiar el resto del código
COPY . .

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

#### 4.1.2. `frontend/Dockerfile`

Este Dockerfile construye la imagen Docker para el servicio frontend. Utiliza un enfoque de construcción multi-etapa para optimizar el tamaño de la imagen final:

**Etapa 1: `builder`**

1.  Se basa en una imagen de Node.js 18.
2.  Establece el directorio de trabajo a `/app`.
3.  Copia los archivos de dependencias (`package.json`, `package-lock.json`) e instala las dependencias de Node.js, incluyendo algunas con `legacy-peer-deps` para compatibilidad.
4.  Copia el resto del código del proyecto.
5.  Define la variable de entorno `NEXT_PUBLIC_BACKEND_URL` en tiempo de construcción.
6.  Ejecuta `npm run build` para compilar la aplicación Next.js.

**Etapa 2: `runner` (Producción)**

1.  Se basa en una imagen de Node.js 18-alpine, que es más ligera.
2.  Crea el directorio de trabajo `/app`.
3.  Copia los archivos de construcción necesarios (públicos, `.next`, `node_modules`, `package.json`) desde la etapa `builder`.
4.  Expone el puerto 3000.
5.  Define el comando de inicio para ejecutar la aplicación Next.js en modo de producción.

```dockerfile
# Etapa 1: Build
FROM node:18 AS builder

# Set working directory
WORKDIR /app

# Copiar archivos de dependencias
COPY package.json package-lock.json* ./

# Instalación con legacy-peer-deps
RUN npm install --legacy-peer-deps \
    && npm install react-markdown remark-gfm rehype-raw --legacy-peer-deps\
    && npm install react-syntax-highlighter @types/react-syntax-highlighter --legacy-peer-deps

# Copiar el resto del proyecto
COPY . .
# Build del proyecto Next.js
ARG NEXT_PUBLIC_BACKEND_URL
ENV NEXT_PUBLIC_BACKEND_URL=$NEXT_PUBLIC_BACKEND_URL

RUN npm run build

# Etapa 2: Producción
FROM node:18-alpine AS runner

# Crear directorio de trabajo
WORKDIR /app

# Copiar archivos necesarios desde el build
COPY --from=builder /app/public ./public
COPY --from=builder /app/.next ./.next
COPY --from=builder /app/node_modules ./node_modules
COPY --from=builder /app/package.json ./package.json

# Exponer el puerto por defecto de Next.js
EXPOSE 3000

# Comando de inicio
CMD ["npx", "next", "start", "-H", "0.0.0.0"]
```

### 4.2. `docker-compose.yml`

El archivo `docker-compose.yml` orquesta los tres servicios del proyecto: `backend`, `frontend` y `postgres`. Permite levantar y gestionar todos los servicios con un solo comando.

*   **`backend`**: Construye la imagen desde el Dockerfile en `./backend`, mapea el puerto 8000, carga variables de entorno desde `./backend/.env`, depende del servicio `postgres` y monta el volumen del código fuente para desarrollo.
*   **`frontend`**: Construye la imagen desde el Dockerfile en `./frontend`, mapea el puerto 3000 y depende del servicio `backend`.
*   **`postgres`**: Utiliza la imagen oficial de PostgreSQL 15, configura variables de entorno para la base de datos, mapea el puerto 5432 y utiliza un volumen persistente para los datos de la base de datos.

Todos los servicios están conectados a una red Docker interna llamada `app-network`.

Para ejecutar el proyecto localmente con Docker Compose, asegúrate de tener los archivos `.env` y `.env.local` configurados correctamente y luego ejecuta:

```bash
docker-compose up --build
```

Esto construirá las imágenes (si es necesario) y levantará los contenedores para el backend, frontend y la base de datos PostgreSQL.

```yaml
version: "3.9"

services:
  backend:
    build:
      context: ./backend
    container_name: backend
    ports:
      - "8000:8000"
    env_file:
      - ./backend/.env
    depends_on:
      - postgres
    volumes:
      - ./backend:/app
    networks:
      - app-network

  frontend:
    build:
      context: ./frontend
    container_name: frontend
    ports:
      - "3000:3000"
    depends_on:
      - backend
    networks:
      - app-network

  postgres:
    image: postgres:15
    container_name: postgres
    restart: always
    environment:
      POSTGRES_USER: postgres_user
      POSTGRES_PASSWORD: postgres_password
      POSTGRES_DB: colabo_automation
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
    networks:
      - app-network

volumes:
  postgres_data:

networks:
  app-network:
```

### 4.3. `render.yaml`

El archivo `render.yaml` es un archivo de configuración para la plataforma de despliegue continuo Render.com. Permite definir la infraestructura y los servicios que se desplegarán en Render, incluyendo bases de datos y servicios web.

En este archivo se configuran:

*   **Bases de datos**: Se define una base de datos PostgreSQL (`colabo-prod-db`) que será utilizada por el backend.
*   **Servicio Backend**: Se configura el despliegue del servicio `backend` como un servicio web Docker. Se especifica la ruta al Dockerfile y al contexto de construcción. Las variables de entorno se configuran aquí, algunas sincronizadas desde Render (como `DB_URL` que se inyecta desde la base de datos de Render) y otras que deben ser configuradas manualmente en el panel de Render (como `CLIENT_ID`, `CLIENT_SECRET`, etc.).
*   **Servicio Frontend**: Se configura el despliegue del servicio `frontend` también como un servicio web Docker, con su Dockerfile y contexto. La variable `NEXT_PUBLIC_BACKEND_URL` se establece para apuntar a la URL del servicio backend desplegado en Render.

Este archivo automatiza el proceso de despliegue en Render, permitiendo un despliegue continuo cada vez que se realizan cambios en el repositorio.

```yaml
databases:
  - name: colabo-prod-db
    databaseName: colabo_automation
    user: postgres_user
    plan: free

services:
  # -------- BACKEND (FastAPI) ----------
  - type: web
    name: backend
    env: docker
    plan: free
    dockerfilePath: backend/Dockerfile
    dockerContext: backend 
    autoDeploy: true
    envVars:
      - key: DB_URL
        fromDatabase:
          name: colabo-prod-db
          property: connectionString
      - key: CLIENT_ID
        sync: false
      - key: CLIENT_SECRET
        sync: false
      - key: REDIRECT_URI
        value: https://backend-15an.onrender.com/auth/callback
      - key: APP_SECRET_KEY
        sync: false
      - key: HASH_SECRET_KEY
        sync: false
      - key: DRIVE_FOLDER_NAME
        value: ColaboAutomation
      - key: COLAB_URL
        value: https://colab.research.google.com/drive/XYZ
      - key: NOTEBOOK_ID
        value: XYZ
      - key: FRONTEND_URL
        value: https://frontend-uwik.onrender.com

  # -------- FRONTEND (Next.js) ----------
  - type: web
    name: frontend
    env: docker
    plan: free
    dockerfilePath: frontend/Dockerfile
    dockerContext: frontend           
    autoDeploy: true
    envVars:
      - key: NEXT_PUBLIC_BACKEND_URL
        value: https://backend-15an.onrender.com
```

## 5. Cómo Utilizar el Código

### 5.1. Requisitos Previos

*   Docker Desktop (o Docker Engine y Docker Compose) instalado.
*   Node.js y npm (para desarrollo frontend sin Docker).
*   Python 3.10 y pip (para desarrollo backend sin Docker).

### 5.2. Configuración Local

1.  **Clonar el repositorio**: Si aún no lo has hecho, clona el repositorio del proyecto.

    ```bash
    git clone <url_del_repositorio>
    cd <nombre_del_repositorio>
    ```

2.  **Crear archivos `.env`**: Crea un archivo `.env` en el directorio `backend/` y un archivo `.env.local` en el directorio `frontend/` con las variables de entorno especificadas en la Sección 3.

### 5.3. Ejecución con Docker Compose

Desde el directorio raíz del proyecto, ejecuta el siguiente comando para construir las imágenes y levantar los servicios:

```bash
docker-compose up --build
```

Una vez que los contenedores estén en funcionamiento, el backend estará accesible en `http://localhost:8000` y el frontend en `http://localhost:3000`.

### 5.4. Despliegue en Render.com

Para desplegar el proyecto en Render.com, sigue estos pasos:

1.  **Conectar tu repositorio**: En el panel de control de Render, crea un nuevo Blueprint y conecta tu repositorio de Git.
2.  **Configurar variables de entorno**: Asegúrate de configurar todas las variables de entorno necesarias (especialmente las marcadas con `sync: false` en `render.yaml`) en el panel de Render para tus servicios backend y frontend.
3.  **Desplegar**: Render detectará el archivo `render.yaml` y configurará automáticamente los servicios y la base de datos. El despliegue se iniciará automáticamente.

## 6. Contribución

Para contribuir a este proyecto, por favor, sigue las siguientes directrices:

*   Crea una rama nueva para cada característica o corrección de errores.
*   Asegúrate de que tu código cumpla con los estándares de calidad y estilo del proyecto.
*   Escribe pruebas unitarias y de integración para tu código.
*   Documenta cualquier cambio significativo en la funcionalidad.



## 7. Rutas del Backend (API Endpoints)

A continuación, se describen las principales rutas de la API del backend, agrupadas por funcionalidad.

### 7.1. Rutas de Autenticación (`/auth`)

Estas rutas gestionan el proceso de autenticación de usuarios utilizando Google OAuth.

*   **`GET /login`**: Inicia el flujo de autenticación OAuth con Google. Redirige al usuario a la página de inicio de sesión de Google.
*   **`GET /callback`**: Endpoint de callback al que Google redirige después de una autenticación exitosa. Procesa el token de Google, crea o actualiza el usuario en la base de datos, genera un token JWT interno y redirige al frontend con este token.
*   **`GET /logout`**: Cierra la sesión del usuario, eliminando la información de sesión.
*   **`GET /user`**: Devuelve la información del usuario actualmente autenticado. Requiere un token JWT válido.

### 7.2. Rutas de Google Drive (`/google_drive`)

Estas rutas interactúan con la API de Google Drive para gestionar notebooks.

*   **`POST /upload_notebook`**: Sube un archivo de notebook (`.ipynb`) especificado a una carpeta en Google Drive del usuario autenticado. Requiere el path del notebook como dato de formulario.
    *   **Parámetros (form data)**:
        *   `notebook_path` (str): Ruta local al archivo del notebook a subir.
    *   **Respuesta**: JSON con el ID del archivo en Drive, un enlace a Colab y el estado.
*   **`GET /list_notebooks`**: Lista los notebooks (`.ipynb`) almacenados en la carpeta de la aplicación en Google Drive del usuario autenticado.
    *   **Respuesta**: JSON con una lista de notebooks, incluyendo su ID, nombre, fecha de creación y enlace a Colab.
*   **`POST /create_and_upload_notebook/{model_id}`**: Crea un notebook para un modelo de Hugging Face específico, lo guarda temporalmente y luego lo sube a Google Drive. Opcionalmente, puede recibir un archivo de pesos específico.
    *   **Parámetros (path)**:
        *   `model_id` (str): El ID del modelo de Hugging Face (ej. `openai-gpt`).
    *   **Parámetros (query)**:
        *   `selected_weight_file` (str, opcional): Nombre del archivo de pesos específico a utilizar.
    *   **Respuesta**: JSON con el ID del archivo en Drive, el enlace a Colab, el ID del modelo y el estado.

### 7.3. Rutas de Hugging Face (`/huggingface`)

Estas rutas interactúan con la API de Hugging Face para obtener información sobre modelos y generar notebooks.

*   **`GET /detalles_modelo/{model_id}`**: Obtiene y devuelve los detalles de los archivos de un modelo específico de Hugging Face, incluyendo el tamaño de los archivos, variantes, recomendaciones y compatibilidad con Colab. Agrupa los archivos `shard`.
    *   **Parámetros (path)**:
        *   `model_id` (str): El ID del modelo de Hugging Face.
    *   **Respuesta**: JSON con una lista de archivos y sus metadatos analizados.
*   **`GET /obtener_readme/{model_id}`**: Descarga y devuelve el contenido del archivo `README.md` de un modelo de Hugging Face.
    *   **Parámetros (path)**:
        *   `model_id` (str): El ID del modelo de Hugging Face.
    *   **Respuesta**: Contenido del README en formato Markdown o un error 404 si no se encuentra.
*   **`GET /route_model/{model_id}`**: Ruta principal para iniciar la generación de un notebook para un modelo. Devuelve una URL para descargar el notebook generado.
    *   **Parámetros (path)**:
        *   `model_id` (str): El ID del modelo de Hugging Face.
    *   **Respuesta**: JSON con el estado, ID del modelo y la URL del endpoint para generar el notebook.
*   **`GET /genera_notebook/{model_id}`**: Genera un notebook (`.ipynb`) para el modelo especificado y lo devuelve como un archivo para descargar. Puede recibir un archivo de pesos específico.
    *   **Parámetros (path)**:
        *   `model_id` (str): El ID del modelo de Hugging Face.
    *   **Parámetros (query)**:
        *   `archivo` (str, opcional): Nombre del archivo de pesos específico a utilizar.
    *   **Respuesta**: Un archivo `.ipynb` para descargar.
*   **`GET /get_model_extension_files`**: Devuelve una lista de las extensiones de archivo de modelo soportadas (ej. `gguf`, `onnx`).
    *   **Respuesta**: JSON con una lista de strings de extensiones.
*   **`GET /buscar_modelos`**: Realiza una búsqueda paginada de modelos en Hugging Face. Permite filtrar por nombre, categoría, pipeline y tags. Excluye modelos ya vistos.
    *   **Parámetros (query)**:
        *   `nombre` (str, opcional): Nombre o parte del nombre del modelo.
        *   `categoria` (str, opcional): Categoría del modelo (ej. `nlp`, `cv`).
        *   `pipeline` (str, opcional): Pipeline del modelo (ej. `text-generation`).
        *   `tags` (List[str], opcional): Lista de tags.
        *   `limit` (int, opcional, default=21): Número máximo de resultados a devolver.
        *   `offset` (int, opcional, default=0): Número de resultados a saltar (para paginación).
        *   `vistos` (List[str], opcional): Lista de IDs de modelos ya vistos para excluir.
    *   **Respuesta**: JSON con los resultados de la búsqueda, información de paginación (`has_more`, `next_offset`).
*   **`GET /categorias_disponibles`**: Devuelve un diccionario con las categorías generales de modelos (nlp, cv, audio, multimodal) y los pipelines específicos asociados a cada una.
    *   **Respuesta**: JSON con un mapeo de categorías a listas de pipelines.

### 7.4. Rutas de Usuario (`/users`)

Estas rutas gestionan las operaciones CRUD para los usuarios y la gestión de sus tokens de Hugging Face.

*   **`GET /me`**: Devuelve la información del usuario actualmente autenticado.
    *   **Respuesta**: Objeto `UserOut` con los datos del usuario.
*   **`GET /{user_id}`**: Obtiene la información de un usuario específico por su ID.
    *   **Parámetros (path)**:
        *   `user_id` (int): ID del usuario.
    *   **Respuesta**: Objeto `UserOut` o error 404 si no se encuentra.
*   **`GET /`**: Lista los usuarios con paginación.
    *   **Parámetros (query)**:
        *   `skip` (int, opcional, default=0): Número de usuarios a saltar.
        *   `limit` (int, opcional, default=10): Número máximo de usuarios a devolver.
    *   **Respuesta**: Lista de objetos `UserOut`.
*   **`PUT /{user_id}`**: Actualiza la información de un usuario específico.
    *   **Parámetros (path)**:
        *   `user_id` (int): ID del usuario.
    *   **Body (JSON)**: Objeto `UserUpdate` con los datos a actualizar.
    *   **Respuesta**: Objeto `UserOut` actualizado o error 404.
*   **`DELETE /{user_id}`**: Elimina un usuario específico.
    *   **Parámetros (path)**:
        *   `user_id` (int): ID del usuario.
    *   **Respuesta**: Objeto `UserOut` del usuario eliminado o error 404.
*   **`GET /me/hf-token`**: Obtiene el token de Hugging Face del usuario actualmente autenticado.
    *   **Respuesta**: String con el token de Hugging Face o `null`.
*   **`POST /me/hf-token`**: Establece o actualiza el token de Hugging Face para el usuario actualmente autenticado.
    *   **Parámetros (form data)**:
        *   `hf_token` (str): El token de Hugging Face a guardar.
    *   **Respuesta**: String con el token de Hugging Face guardado.
*   **`PUT /me/hf-token`**: (Alias de POST) Establece o actualiza el token de Hugging Face para el usuario actualmente autenticado.
    *   **Parámetros (form data)**:
        *   `hf_token` (str): El token de Hugging Face a guardar.
    *   **Respuesta**: String con el token de Hugging Face guardado.




