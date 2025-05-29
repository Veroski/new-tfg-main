"use client"

import { useState, useEffect, useCallback, useRef } from "react"
import { Search, Loader2 } from "lucide-react"
import { Input } from "@/components/ui/input"
import { Card, CardContent } from "@/components/ui/card"
import Link from "next/link"
import { filtrarModelos } from "@/lib/api"

const TAREA_POR_DEFECTO = "text-generation"
const MAX_RESULTADOS = 21
const DEBOUNCE_MS = 300

const CATEGORIAS = [
  { key: "general", label: "General" },
  { key: "texto", label: "Texto" },
  { key: "imagen", label: "Imagen" },
  { key: "audio", label: "Audio" },
  { key: "video", label: "Vídeo" },
]

type ModeloUI = {
  id: string
  name: string
  description: string
  tags: string[]
}

export default function ExplorarModelos() {
  const [searchTerm, setSearchTerm] = useState("")
  const [debouncedSearch, setDebouncedSearch] = useState("")
  const [models, setModels] = useState<ModeloUI[]>([])
  const [loading, setLoading] = useState(false)
  const [isFetchingMore, setIsFetchingMore] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [hasMore, setHasMore] = useState(false)
  const [offset, setOffset] = useState(0)
  const [reset, setReset] = useState(false)
  const [categoria, setCategoria] = useState<"general" | "texto" | "imagen" | "audio" | "video">("texto")
  
  const idsVistos = useRef<Set<string>>(new Set())
  const scrollContainerRef = useRef<HTMLDivElement | null>(null)
  const sentinelRef = useRef<HTMLDivElement | null>(null)
  const abortRef = useRef<AbortController | null>(null)

  useEffect(() => {
    const id = setTimeout(() => setDebouncedSearch(searchTerm), DEBOUNCE_MS)
    return () => clearTimeout(id)
  }, [searchTerm])

  useEffect(() => {
    idsVistos.current.clear()
    setModels([])
    setOffset(0)
    setHasMore(false)
    setReset(true)
  }, [debouncedSearch, categoria])

  const fetchModels = useCallback(
    async (offsetParam: number) => {
      if (abortRef.current) abortRef.current.abort()
      abortRef.current = new AbortController()

      const isFirst = offsetParam === 0
      if (isFirst) setLoading(true)
      else setIsFetchingMore(true)

      try {
        setError(null)
        const seenArray = Array.from(idsVistos.current)

        const url = filtrarModelos(
          TAREA_POR_DEFECTO,
          categoria,
          debouncedSearch,
          "downloads",
          MAX_RESULTADOS,
          offsetParam,
          seenArray              
        )

        const res = await fetch(url, { signal: abortRef.current.signal })
        if (!res.ok) throw new Error(`Error ${res.status}`)

        const data = await res.json()

        const nuevos = data.results.filter(
          (m: any) => !idsVistos.current.has(m.nombre)
        )
        nuevos.forEach((m: any) => idsVistos.current.add(m.nombre))

        const adaptados: ModeloUI[] = nuevos.map((item: any) => ({
          id: item.nombre,
          name: item.nombre,
          description: `${item.descargas?.toLocaleString() ?? "0"} descargas`,
          tags: item.tags ?? [],
        }))

        setModels(prev => (isFirst ? adaptados : [...prev, ...adaptados]))
        setHasMore(data.has_more)
        if (data.has_more) setOffset(data.next_offset)
      } catch (e: any) {
        if (e.name !== "AbortError") setError(e.message)
      } finally {
        if (isFirst) setLoading(false)
        else setIsFetchingMore(false)
      }
    },
    [debouncedSearch, categoria]
  )

  useEffect(() => {
    if (reset) {
      setReset(false)
      fetchModels(0)
    }
  }, [reset, fetchModels])

  useEffect(() => {
    const container = scrollContainerRef.current
    const sentinel = sentinelRef.current
    if (!hasMore || !container || !sentinel || isFetchingMore) return

    let triggered = false
    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting && !triggered) {
          triggered = true
          fetchModels(offset)
        }
      },
      { root: container, threshold: 1, rootMargin: "120px" }
    )

    observer.observe(sentinel)
    return () => observer.disconnect()
  }, [hasMore, isFetchingMore, offset, fetchModels])

  return (
    <div className="container mx-auto px-4 py-8 h-screen flex flex-col">
      <h1 className="text-2xl md:text-3xl font-bold mb-4">Explorador de Modelos</h1>

      <div className="mb-4 flex gap-4 border-b">
        {CATEGORIAS.map(tab => (
          <button
            key={tab.key}
            onClick={() => setCategoria(tab.key as any)}
            className={`pb-2 px-1 text-sm md:text-base ${categoria === tab.key
                ? "border-b-2 border-[#FB923C] text-[#FB923C]"
                : "text-gray-500 hover:text-[#FB923C]"
              }`}
          >
            {tab.label}
          </button>
        ))}
      </div>

      <div className="flex flex-1 gap-6 overflow-hidden">
        <div className="flex-1 flex flex-col overflow-hidden">
          <div className="relative mb-4">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-400" />
            <Input
              placeholder="Buscar modelos"
              className="pl-10"
              value={searchTerm}
              onChange={e => setSearchTerm(e.target.value)}
            />
          </div>

          <div ref={scrollContainerRef} className="flex-1 overflow-y-auto pr-1 space-y-6">
            {loading && (
              <p className="text-center text-gray-500 py-12">Cargando modelos…</p>
            )}

            {error && (
              <p className="text-center text-red-500 py-12">{error}. Intenta de nuevo.</p>
            )}

            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
              {models.map(model => (
                <Link
                  key={model.id}
                  href={`/modelo/${encodeURIComponent(model.id)}`}
                  className="animate-fadeIn"
                >
                  <Card className="h-full hover:shadow-md transition-shadow border-gray-200 hover:border-[#FB923C]">
                    <CardContent className="p-4">
                      <div className="w-12 h-12 bg-orange-100 rounded-full flex items-center justify-center mb-4">
                        <Search className="h-6 w-6 text-[#F97316]" />
                      </div>
                      <h3 className="font-semibold text-lg mb-2">{model.name}</h3>
                      <p className="text-gray-600 text-sm mb-3">{model.description}</p>
                      <div className="flex flex-wrap gap-2">
                        {model.tags.map(tag => (
                          <span
                            key={`${model.id}-${tag}`}
                            className="inline-block bg-orange-100 text-[#F97316] text-xs px-2 py-1 rounded-full"
                          >
                            {tag}
                          </span>
                        ))}
                      </div>
                    </CardContent>
                  </Card>
                </Link>
              ))}
            </div>

            {hasMore && (
              <>
                <div ref={sentinelRef} className="h-10 mt-4" />
                {isFetchingMore && (
                  <div className="flex justify-center mt-4">
                    <Loader2 className="h-5 w-5 text-[#F97316] animate-spin" />
                  </div>
                )}
              </>
            )}

            {!hasMore && models.length > 0 && !loading && (
              <p className="text-center text-gray-400 mt-6">No hay más modelos.</p>
            )}

            {models.length === 0 && !loading && (
              <div className="text-center py-12">
                <p className="text-gray-500">
                  No se encontraron modelos con los criterios seleccionados.
                </p>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}
