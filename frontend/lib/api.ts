// api.ts o routes.ts actualizado

export const API_BASE_URL = process.env.BACKEND_URL;

export const filtrarModelos = (
  task: string,
  categoria: string,
  q: string,
  sort: string,
  pageSize: number,
  offset: number,
  seenIds: string[]        // << nuevo
) => {
  const p = new URLSearchParams({
    tarea: task,
    categoria,
    q,
    sort,
    page_size: String(pageSize),
    offset: String(offset),
  })
  if (seenIds.length) p.set("seen", seenIds.join(","))
  return `${API_BASE_URL}/huggingface/filtrar_modelos?${p.toString()}`
}


// Ruta para obtener detalles de un modelo específico
export const obtenerDetallesModelo = (modelId: string): string => {
  return `${API_BASE_URL}/huggingface/detalles_modelo/${encodeURIComponent(modelId)}`;
};

// Ruta para obtener el README.md de un modelo
export const obtenerReadmeModelo = (modelId: string): string => {
  return `${API_BASE_URL}/huggingface/obtener_readme/${encodeURIComponent(modelId)}`;
};

// Ruta para generar un notebook para un modelo
export const generarNotebook = (modelId: string): string => {
  return `${API_BASE_URL}/huggingface/genera_notebook/${encodeURIComponent(modelId)}`;
};

// Ruta para crear y subir un notebook directamente a Google Colab
export const crearYSubirNotebook = (modelId: string): string => {
  return `${API_BASE_URL}/google_drive/create_and_upload_notebook/${encodeURIComponent(modelId)}`;
};

// Ruta para verificar el estado de autenticación del usuario
export const verificarAutenticacion = (): string => {
  return `${API_BASE_URL}/auth/user`;
};

// Ruta para iniciar el proceso de login
export const iniciarLogin = (): string => {
  return `${API_BASE_URL}/auth/login`;
};

export const buscarModelosLive = (
  query: string,
  page = 1,
  pageSize = 10,
  tarea = "text-generation",
  incluir_tags?: string[]
) => {
  const params = new URLSearchParams({
    q: query,
    tarea,
    page: String(page),
    page_size: String(pageSize),
  })
  incluir_tags?.forEach(t => params.append("incluir_tags", t))
  return `${API_BASE_URL}/huggingface/buscar_modelos_live?${params.toString()}`
}

export const routeModel = (modelId: string): string => {
  // 🆕  único endpoint que construimos en el backend
  return `${API_BASE_URL}/huggingface/route_model/${encodeURIComponent(modelId)}`;
};

// 🆕 Ruta para obtener el usuario actual (autenticado)
export const getUsuarioActual = (): string => {
  return `${API_BASE_URL}/users/me`;
};
