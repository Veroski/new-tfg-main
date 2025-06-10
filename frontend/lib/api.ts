// api.ts o routes.ts actualizado

export const API_BASE_URL = process.env.NEXT_PUBLIC_BACKEND_URL;



// Ruta para obtener detalles de un modelo específico
export const obtenerDetallesModelo = (modelId: string): string => {
  return `${API_BASE_URL}/huggingface/detalles_modelo/${encodeURIComponent(modelId)}`;
};

// Ruta para obtener el README.md de un modelo
export const obtenerReadmeModelo = (modelId: string): string => {
  return `${API_BASE_URL}/huggingface/obtener_readme/${encodeURIComponent(modelId)}`;
};

// Ruta para generar un notebook para un modelo
export const generarNotebook = async (modelId: string): Promise<string> => {
  const res = await fetch(`${API_BASE_URL}/huggingface/genera_notebook/${encodeURIComponent(modelId)}`, {
    method: "GET",
    headers: {
      "Content-Type": "application/json",
      "Authorization": `Bearer ${localStorage.getItem("auth_token")}`, 
    },
  });
  if (!res.ok) {
    throw new Error(`Error ${res.status} al generar el notebook`);
  }
  return await res.text();
};

// Ruta para crear y subir un notebook directamente a Google Colab
export const crearYSubirNotebook = (modelId: string, selectedWeightFile?: string): string => {
  const base = `${API_BASE_URL}/google_drive/create_and_upload_notebook/${encodeURIComponent(modelId)}`
  return selectedWeightFile
    ? `${base}?selected_weight_file=${encodeURIComponent(selectedWeightFile)}`
    : base
}


// Ruta para verificar el estado de autenticación del usuario
export const verificarAutenticacion = (): string => {
  return `${API_BASE_URL}/auth/user`;
};

// Ruta para iniciar el proceso de login
export const iniciarLogin = (): string => {
  return `${API_BASE_URL}/auth/login`;
};


export const routeModel = (modelId: string): string => {
  // 🆕  único endpoint que construimos en el backend
  return `${API_BASE_URL}/huggingface/route_model/${encodeURIComponent(modelId)}`;
};


export const modelExtensionFile = async (): Promise<string[]> => {
  // Llama al endpoint del backend que devuelve las extensiones
  const res = await fetch(`${API_BASE_URL}/huggingface/get_model_extension_files`, {
    method: "GET",
    headers: {
      "Content-Type": "application/json",
    },
  });

  if (!res.ok) {
    throw new Error(`Error ${res.status} al obtener extensiones de modelo`);
  }

  const data = await res.json();

  // Validamos que sea una lista de strings como [".gguf", ".onnx", ...]
  if (!Array.isArray(data) || !data.every((ext) => typeof ext === "string")) {
    throw new Error("Respuesta inválida: se esperaba una lista de strings");
  }

  return data;
};

// 🆕 Ruta para obtener el usuario actual (autenticado)
export const getUsuarioActual = (): string => {
  return `${API_BASE_URL}/users/me`;
};


// 🧠 Hugging Face token – obtener el token del usuario actual
export const getHfToken = async (): Promise<string | null> => {
  const token = localStorage.getItem("auth_token");
  const res = await fetch(`${API_BASE_URL}/users/me/hf-token`, {
    method: "GET",
    credentials: "include",
    headers: {
      "Authorization": `Bearer ${token}`, // Si necesitas autenticación
    },
  });

  if (!res.ok) throw new Error(`Error ${res.status} al obtener el HF token`);
  return await res.text(); // o res.json() si lo cambias en backend
};

// 🧠 Hugging Face token – establecer el token del usuario actual (POST)
export const postHfToken = async (hf_token: string): Promise<string> => {
  const token = localStorage.getItem("auth_token");
  const res = await fetch(`${API_BASE_URL}/users/me/hf-token`, {
    method: "POST",
    headers: {
      "Content-Type": "application/x-www-form-urlencoded",
      "Authorization": `Bearer ${token}`, // Si necesitas autenticación
    },
    body: new URLSearchParams({ hf_token }),
    credentials: "include",
  });

  if (!res.ok) throw new Error(`Error ${res.status} al guardar el HF token`);
  return await res.text();
};

// 🧠 Hugging Face token – actualizar el token del usuario actual (PUT)
export const putHfToken = async (hf_token: string): Promise<string> => {
  const token = localStorage.getItem("auth_token");
  const res = await fetch(`${API_BASE_URL}/users/me/hf-token`, {
    method: "PUT",
    headers: {
      "Content-Type": "application/x-www-form-urlencoded",
      "Authorization": `Bearer ${token}`, // Si necesitas autenticación
    },
    body: new URLSearchParams({ hf_token }),
    credentials: "include",
  });

  if (!res.ok) throw new Error(`Error ${res.status} al actualizar el HF token`);
  return await res.text();
};


export const filtrarModelos = (
  tarea: string | null,
  categoria: string | null,
  nombre: string,
  orden: string,
  limit: number,
  offset: number,
  vistos: string[]
): string => {
  const params = new URLSearchParams()

  if (nombre) params.append("nombre", nombre)
  if (categoria) params.append("categoria", categoria)
  if (tarea) params.append("pipeline", tarea)
  if (vistos.length) vistos.forEach(v => params.append("vistos", v))
  params.append("limit", limit.toString())
  params.append("offset", offset.toString())

  return `${API_BASE_URL}/huggingface/buscar_modelos?${params.toString()}`
}


// 🆕 Ruta para obtener todas las categorías y pipelines disponibles
export const obtenerCategoriasDisponibles = async (): Promise<Record<string, string[]>> => {
  const res = await fetch(`${API_BASE_URL}/huggingface/categorias_disponibles`, {
    method: "GET",
    headers: {
      "Content-Type": "application/json",
      "Authorization": `Bearer ${localStorage.getItem("auth_token")}`, 
    },
  });

  if (!res.ok) {
    throw new Error(`Error ${res.status} al obtener categorías y pipelines`);
  }

  return await res.json(); // { nlp: ["text-generation", ...], cv: ["image-classification", ...], ... }
};
