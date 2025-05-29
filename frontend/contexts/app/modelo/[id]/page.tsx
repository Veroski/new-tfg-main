"use client"

import remarkGfm from "remark-gfm"
import rehypeRaw from "rehype-raw"
import { useEffect, useState, useCallback } from "react"
import { useParams, useRouter } from "next/navigation"
import { ArrowLeft, Tag, Star, NotebookPen, FileText, ExternalLink } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Card, CardContent } from "@/components/ui/card"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert"
import {
  obtenerDetallesModelo,
  obtenerReadmeModelo,
  generarNotebook,
  crearYSubirNotebook,
  verificarAutenticacion,
  iniciarLogin,
} from "@/lib/api"
import ReactMarkdown from "react-markdown"
import { Prism as SyntaxHighlighter } from "react-syntax-highlighter"
import { oneDark } from "react-syntax-highlighter/dist/esm/styles/prism"

/* --------- Tipos que devuelve el backend --------- */
type Archivo = {
  archivo: string
  tamaño_gb: number | null
  tamaño_bytes: number | null
  variant: string
  rank: number
  recomendacion: string
  colab_status: "✅" | "⚠️" | "❌" | "?"
  colab_msg: string
}

type ModeloDetalles = {
  id: string
  nombre?: string
  descargas?: number
  ultima_modificacion?: string
  tags?: string[]
  archivos: Archivo[]
}

export default function ModeloDetalle() {
  const router = useRouter()
  const { id: rawId } = useParams<{ id: string }>()
  const modelId = decodeURIComponent(rawId)

  const [data, setData] = useState<ModeloDetalles | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [readme, setReadme] = useState<string | null>(null)
  const [readmeLoading, setReadmeLoading] = useState(true)
  const [readmeError, setReadmeError] = useState<string | null>(null)
  const [isAuthenticated, setIsAuthenticated] = useState<boolean>(false)
  const [authChecking, setAuthChecking] = useState<boolean>(true)
  const [colabLink, setColabLink] = useState<string | null>(null)
  const [creatingColab, setCreatingColab] = useState<boolean>(false)

  /* -------- Verificar autenticación del usuario -------- */
  const checkAuth = useCallback(async () => {
    try {
      setAuthChecking(true)
      const res = await fetch(verificarAutenticacion(), {
        credentials: "include",
      })
      setIsAuthenticated(res.ok)
    } catch (err) {
      setIsAuthenticated(false)
    } finally {
      setAuthChecking(false)
    }
  }, [])

  /* -------- Llamada a la API para detalles del modelo -------- */
  const fetchDetails = useCallback(async () => {
    try {
      setLoading(true)
      setError(null)

      const res = await fetch(obtenerDetallesModelo(modelId))
      if (!res.ok) throw new Error(`Error ${res.status}`)

      const json = await res.json()
      if (Array.isArray(json)) {
        setData({
          id: modelId,
          archivos: json,
          nombre: modelId,
          descargas: undefined,
          ultima_modificacion: undefined,
          tags: [],
        })
      } else {
        setData({
          id: modelId,
          nombre: json.nombre ?? modelId,
          descargas: json.descargas,
          ultima_modificacion: json.ultima_modificacion,
          tags: json.tags ?? [],
          archivos: json.archivos ?? [],
        })
      }
    } catch (err: any) {
      setError(err.message ?? "Error desconocido")
    } finally {
      setLoading(false)
    }
  }, [modelId])

  /* -------- Llamada a la API para obtener README -------- */
  const fetchReadme = useCallback(async () => {
    try {
      setReadmeLoading(true)
      setReadmeError(null)

      const res = await fetch(obtenerReadmeModelo(modelId))
      if (!res.ok) {
        if (res.status === 404) {
          setReadmeError("No se encontró el README.md para este modelo")
          return
        }
        throw new Error(`Error ${res.status}`)
      }

      const text = await res.text()
      setReadme(text)
    } catch (err: any) {
      setReadmeError(err.message ?? "Error al cargar el README")
    } finally {
      setReadmeLoading(false)
    }
  }, [modelId])

  useEffect(() => {
    fetchDetails()
    fetchReadme()
    checkAuth()
  }, [fetchDetails, fetchReadme, checkAuth])

  /* -------- Helpers -------- */
  const formatNumber = (n?: number) => n?.toLocaleString("es-ES") ?? "—"

  const formatBytes = (bytes?: number | null) => {
    if (bytes == null) return "—"
    if (bytes < 1024) return `${bytes} B`
    if (bytes < 1_048_576) return `${(bytes / 1024).toFixed(2)} KB`
    if (bytes < 1_073_741_824) return `${(bytes / 1_048_576).toFixed(2)} MB`
    return `${(bytes / 1_073_741_824).toFixed(2)} GB`
  }

  const StarRating = ({ rank, unknown }: { rank: number; unknown: boolean }) => {
    if (unknown) {
      return <span className="text-yellow-400 text-lg">❓</span>
    }
    const filled = Array(rank).fill(0)
    const empty = Array(5 - rank).fill(0)
    return (
      <div className="flex gap-0.5">
        {filled.map((_, i) => (
          <Star key={`f${i}`} className="h-4 w-4 text-yellow-400 fill-yellow-400" />
        ))}
        {empty.map((_, i) => (
          <Star key={`e${i}`} className="h-4 w-4 text-gray-300" />
        ))}
      </div>
    )
  }

  /* -------- Acción: generar notebook -------- */
  const handleGenerateNotebook = async () => {
    try {
      // Redirigir al usuario directamente a la URL de descarga del notebook
      window.open(generarNotebook(modelId), "_blank")
    } catch (err: any) {
      alert("Error al generar notebook: " + (err?.message ?? "desconocido"))
    }
  }

  /* -------- Acción: crear y subir notebook a Colab -------- */
  const handleCreateColabNotebook = async () => {
    if (!isAuthenticated) {
      // Redirigir al usuario a la página de login
      window.location.href = iniciarLogin()
      return
    }

    try {
      setCreatingColab(true)
      setColabLink(null)

      const res = await fetch(crearYSubirNotebook(modelId), {
        method: "POST",
        credentials: "include",
        headers: {
          "Content-Type": "application/json",
        },
      })

      if (!res.ok) {
        throw new Error(`Error ${res.status}: ${await res.text()}`)
      }

      const data = await res.json()
      setColabLink(data.colab_link)
    } catch (err: any) {
      alert("Error al crear notebook en Colab: " + (err?.message ?? "desconocido"))
    } finally {
      setCreatingColab(false)
    }
  }

  /* -------- UI -------- */
  if (loading)
    return (
      <div className="container mx-auto px-4 py-12 text-center">
        <p className="text-gray-500">Cargando modelo…</p>
      </div>
    )

  if (error || !data)
    return (
      <div className="container mx-auto px-4 py-12 text-center">
        <h1 className="text-2xl font-bold mb-4">{error ? "Error al cargar el modelo" : "Modelo no encontrado"}</h1>
        <Button onClick={() => router.push("/explorar")}>Volver al explorador</Button>
      </div>
    )

  return (
    <div className="container mx-auto px-4 py-8">
      <Button variant="ghost" className="mb-6 flex items-center gap-2" onClick={() => router.push("/explorar")}>
        <ArrowLeft className="h-4 w-4" />
        Volver al explorador
      </Button>

      <Card>
        <CardContent className="p-6">
          <div className="flex flex-col md:flex-row md:justify-between md:items-center gap-4 mb-6">
            <h1 className="text-2xl md:text-3xl font-bold">{data.nombre ?? data.id}</h1>
            <div className="flex flex-col sm:flex-row gap-2">
              <Button onClick={handleGenerateNotebook} className="bg-[#F97316] hover:bg-[#EA580C]">
                <NotebookPen className="h-4 w-4 mr-2" />
                Descargar Notebook
              </Button>
              <Button
                onClick={handleCreateColabNotebook}
                disabled={creatingColab || authChecking}
                className="bg-blue-600 hover:bg-blue-700"
              >
                <ExternalLink className="h-4 w-4 mr-2" />
                {creatingColab ? "Creando..." : "Crear en Colab"}
              </Button>
            </div>
          </div>

          {colabLink && (
            <Alert className="mb-6 bg-blue-50 border-blue-200">
              <AlertTitle className="text-blue-800">¡Notebook creado en Google Colab!</AlertTitle>
              <AlertDescription className="text-blue-700">
                <div className="mt-2">
                  <a
                    href={colabLink}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="flex items-center gap-2 text-blue-600 hover:text-blue-800 underline"
                  >
                    <ExternalLink className="h-4 w-4" />
                    Abrir notebook en Google Colab
                  </a>
                </div>
              </AlertDescription>
            </Alert>
          )}

          {(!!data.ultima_modificacion || typeof data.descargas === "number" || (data.tags?.length ?? 0) > 0) && (
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
              {data.ultima_modificacion && (
                <div className="space-y-2">
                  <p className="text-sm text-gray-500">Última modificación</p>
                  <p className="font-medium">{data.ultima_modificacion}</p>
                </div>
              )}

              {typeof data.descargas === "number" && (
                <div className="space-y-2">
                  <p className="text-sm text-gray-500">Descargas</p>
                  <p className="font-medium">{formatNumber(data.descargas)}</p>
                </div>
              )}

              {data.tags && data.tags.length > 0 && (
                <div className="space-y-2">
                  <p className="text-sm text-gray-500">Tags</p>
                  <div className="flex flex-wrap gap-2">
                    {data.tags.map((tag) => (
                      <span
                        key={tag}
                        className="inline-flex items-center gap-1 bg-orange-100 text-[#F97316] px-2 py-1 rounded-full text-sm"
                      >
                        <Tag className="h-3 w-3" />
                        {tag}
                      </span>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}

          <Tabs defaultValue="readme" className="mt-6">
            <TabsList>
              <TabsTrigger value="readme" className="flex items-center gap-2">
                <FileText className="h-4 w-4" />
                README
              </TabsTrigger>
              <TabsTrigger value="archivos" className="flex items-center gap-2">
                <FileText className="h-4 w-4" />
                Archivos
              </TabsTrigger>
            </TabsList>

            <TabsContent value="readme" className="mt-4">
              {readmeLoading ? (
                <div className="py-8 text-center">
                  <p className="text-gray-500">Cargando README.md...</p>
                </div>
              ) : readmeError ? (
                <div className="py-8 text-center">
                  <p className="text-red-500">{readmeError}</p>
                </div>
              ) : (
                <div className="readme-container bg-white rounded-lg border p-6">
                  <ReactMarkdown
                    remarkPlugins={[remarkGfm]}
                    rehypePlugins={[rehypeRaw]}
                    components={{
                      code({ className, children, ...props }) {
                        const match = /language-(\w+)/.exec(className || "")
                        const isInline = !match

                        return isInline ? (
                          <code className={`${className} bg-gray-100 px-1 py-0.5 rounded text-sm font-mono`} {...props}>
                            {children}
                          </code>
                        ) : (
                          <SyntaxHighlighter
                            style={oneDark as any}
                            language={match[1]}
                            PreTag="div"
                            className="rounded-md"
                          >
                            {String(children).replace(/\n$/, "")}
                          </SyntaxHighlighter>

                        )
                      },
                      // Mejorar el manejo de texto plano para caracteres especiales
                      text({ children }) {
                        if (typeof children === "string") {
                          // Manejar caracteres de escape y símbolos especiales
                          return children
                            .replace(/\\\\/g, "\\") // Doble backslash a simple
                            .replace(/\\n/g, "\n") // Newlines escapados
                            .replace(/\\t/g, "\t") // Tabs escapados
                            .replace(/\\"/g, '"') // Comillas escapadas
                            .replace(/\\'/g, "'") // Comillas simples escapadas
                        }
                        return children
                      },
                      h1: ({ children }) => (
                        <h1 className="text-3xl font-bold mb-4 pb-2 border-b border-gray-200 text-gray-900">
                          {children}
                        </h1>
                      ),
                      h2: ({ children }) => (
                        <h2 className="text-2xl font-semibold mb-3 mt-6 pb-2 border-b border-gray-200 text-gray-900">
                          {children}
                        </h2>
                      ),
                      h3: ({ children }) => (
                        <h3 className="text-xl font-semibold mb-2 mt-5 text-gray-900">{children}</h3>
                      ),
                      h4: ({ children }) => (
                        <h4 className="text-lg font-semibold mb-2 mt-4 text-gray-900">{children}</h4>
                      ),
                      h5: ({ children }) => (
                        <h5 className="text-base font-semibold mb-2 mt-3 text-gray-900">{children}</h5>
                      ),
                      h6: ({ children }) => (
                        <h6 className="text-sm font-semibold mb-2 mt-3 text-gray-700">{children}</h6>
                      ),
                      p: ({ children }) => <p className="mb-4 text-gray-700 leading-relaxed">{children}</p>,
                      a: ({ href, children }) => (
                        <a
                          href={href}
                          className="text-blue-600 hover:text-blue-800 underline decoration-blue-600/30 hover:decoration-blue-800/50 transition-colors"
                          target="_blank"
                          rel="noopener noreferrer"
                        >
                          {children}
                        </a>
                      ),
                      ul: ({ children }) => <ul className="mb-4 ml-6 space-y-1 list-disc text-gray-700">{children}</ul>,
                      ol: ({ children }) => (
                        <ol className="mb-4 ml-6 space-y-1 list-decimal text-gray-700">{children}</ol>
                      ),
                      li: ({ children }) => <li className="leading-relaxed">{children}</li>,
                      blockquote: ({ children }) => (
                        <blockquote className="border-l-4 border-gray-300 pl-4 py-2 mb-4 bg-gray-50 text-gray-700 italic">
                          {children}
                        </blockquote>
                      ),
                      table: ({ children }) => (
                        <div className="overflow-x-auto mb-4">
                          <table className="min-w-full border border-gray-300 rounded-lg overflow-hidden">
                            {children}
                          </table>
                        </div>
                      ),
                      thead: ({ children }) => <thead className="bg-gray-50">{children}</thead>,
                      tbody: ({ children }) => <tbody className="divide-y divide-gray-200">{children}</tbody>,
                      tr: ({ children }) => <tr className="hover:bg-gray-50">{children}</tr>,
                      th: ({ children }) => (
                        <th className="px-4 py-2 text-left font-semibold text-gray-900 border-b border-gray-300">
                          {children}
                        </th>
                      ),
                      td: ({ children }) => (
                        <td className="px-4 py-2 text-gray-700 border-b border-gray-200">{children}</td>
                      ),
                      hr: () => <hr className="my-6 border-gray-300" />,
                      strong: ({ children }) => <strong className="font-semibold text-gray-900">{children}</strong>,
                      em: ({ children }) => <em className="italic text-gray-700">{children}</em>,
                      img: ({ src, alt }) => (
                        <img
                          src={src || "/placeholder.svg"}
                          alt={alt}
                          className="max-w-full h-auto rounded-lg shadow-sm mb-4 border border-gray-200"
                        />
                      ),
                      pre: ({ children }) => <div className="mb-4">{children}</div>,
                    }}
                  >
                    {readme ?? ""}
                  </ReactMarkdown>
                </div>
              )}
            </TabsContent>

            <TabsContent value="archivos" className="mt-4">
              {/* -------- Tabla de archivos -------- */}
              <div>
                <h2 className="text-xl font-semibold mb-4">Archivos</h2>
                {data.archivos.length === 0 ? (
                  <p className="text-gray-500">Sin archivos.</p>
                ) : (
                  <div className="overflow-x-auto">
                    <Table>
                      <TableHeader>
                        <TableRow>
                          <TableHead>Archivo</TableHead>
                          <TableHead>Tamaño</TableHead>
                          <TableHead>Colab</TableHead>
                          <TableHead>Precisión</TableHead>
                          <TableHead>Calidad</TableHead>
                        </TableRow>
                      </TableHeader>
                      <TableBody>
                        {data.archivos.map((f, i) => (
                          <TableRow key={i}>
                            <TableCell className="font-medium">{f.archivo}</TableCell>
                            <TableCell>{formatBytes(f.tamaño_bytes)}</TableCell>
                            <TableCell title={f.colab_msg}>{f.colab_status}</TableCell>
                            <TableCell title={f.recomendacion}>
                              {f.variant === "?" ? <span className="text-yellow-400 text-lg">❓</span> : f.variant}
                            </TableCell>
                            <TableCell>
                              <StarRating rank={f.rank} unknown={f.variant === "?"} />
                            </TableCell>
                          </TableRow>
                        ))}
                      </TableBody>
                    </Table>
                  </div>
                )}
              </div>
            </TabsContent>
          </Tabs>
        </CardContent>
      </Card>
    </div>
  )
}
