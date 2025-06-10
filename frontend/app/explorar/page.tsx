"use client"

import { useEffect, useState, useRef, useCallback } from "react"
import { Search, Loader2 } from "lucide-react"
import Link from "next/link"
import { Input } from "@/components/ui/input"
import { Card, CardContent } from "@/components/ui/card"
import { obtenerCategoriasDisponibles, filtrarModelos } from "@/lib/api"

type ModeloUI = {
  id: string
  name: string
  description: string
  tags: string[]
}

const LIMIT = 21

export default function ExplorarModelos() {
  const [categorias, setCategorias] = useState<Record<string, string[]>>({})
  const [categoria, setCategoria] = useState<string>("nlp")
  const [pipeline, setPipeline] = useState<string | null>(null)

  const [searchTerm, setSearchTerm] = useState("")
  const [debouncedSearch, setDebouncedSearch] = useState("")
  const [models, setModels] = useState<ModeloUI[]>([])
  const [offset, setOffset] = useState(0)
  const [hasMore, setHasMore] = useState(true)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [resetTrigger, setResetTrigger] = useState(0)

  // Establecer valor por defecto en búsqueda para categoría nlp
  useEffect(() => {
    if (categoria === "nlp" && searchTerm === "") {
      setSearchTerm("gpt2")
    } else if (categoria !== "nlp" && searchTerm === "gpt2") {
      setSearchTerm("")
    }
  }, [categoria])

  const seenIds = useRef<Set<string>>(new Set())
  const sentinelRef = useRef<HTMLDivElement | null>(null)
  const scrollRef = useRef<HTMLDivElement | null>(null)
  const abortRef = useRef<AbortController | null>(null)

  // Debounce de búsqueda
  useEffect(() => {
    const id = setTimeout(() => setDebouncedSearch(searchTerm), 300)
    return () => clearTimeout(id)
  }, [searchTerm])

  // Cargar categorías iniciales
  useEffect(() => {
    const fetchCategorias = async () => {
      try {
        const data = await obtenerCategoriasDisponibles()
        setCategorias(data)
        const keys = Object.keys(data)
        if (keys.length) setCategoria(keys.includes("nlp") ? "nlp" : keys[0])
      } catch (e) {
        setError("No se pudieron cargar las categorías")
      }
    }
    fetchCategorias()
  }, [])

  // Reiniciar búsqueda al cambiar filtros
  useEffect(() => {
    seenIds.current.clear()
    setModels([])
    setOffset(0)
    setHasMore(true)
    setResetTrigger((prev) => prev + 1)
  }, [debouncedSearch, categoria, pipeline])

  // Buscar modelos paginados
  const fetchModels = useCallback(
    async (offsetParam: number) => {
      if (abortRef.current) abortRef.current.abort()
      abortRef.current = new AbortController()

      setLoading(true)
      setError(null)
      try {
        const url = filtrarModelos(
          pipeline,
          categoria,
          debouncedSearch,
          "downloads",
          LIMIT,
          offsetParam,
          Array.from(seenIds.current),
        )
        const res = await fetch(url, {
          method: "GET",
          signal: abortRef.current.signal,
          headers: {
            "Content-Type": "application/json",
            "Authorization": `Bearer ${localStorage.getItem("auth_token")}`,
          },
        })

        if (!res.ok) throw new Error("Error al obtener modelos")

        const data = await res.json()
        const nuevos = data.results.filter((m: any) => !seenIds.current.has(m.model_id))
        nuevos.forEach((m: any) => seenIds.current.add(m.model_id))

        const adaptados: ModeloUI[] = nuevos.map((m: any) => ({
          id: m.model_id,
          name: m.model_id,
          description: m.private ? "Privado" : "Público",
          tags: m.tags ?? [],
        }))

        setModels((prev) => (offsetParam === 0 ? adaptados : [...prev, ...adaptados]))
        setOffset(data.next_offset)
        setHasMore(data.has_more)
      } catch (err: any) {
        if (err.name !== "AbortError") setError("Error al buscar modelos")
      } finally {
        setLoading(false)
      }
    },
    [categoria, pipeline, debouncedSearch],
  )

  // Trigger de carga
  useEffect(() => {
    fetchModels(0)
  }, [resetTrigger])

  // Scroll infinito con IntersectionObserver
  useEffect(() => {
    const container = scrollRef.current
    const sentinel = sentinelRef.current
    if (!hasMore || !container || !sentinel) return

    let triggered = false
    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting && !triggered) {
          triggered = true
          fetchModels(offset)
        }
      },
      { root: container, threshold: 1, rootMargin: "120px" },
    )

    observer.observe(sentinel)
    return () => observer.disconnect()
  }, [offset, hasMore, fetchModels])

  return (
    <div className="container mx-auto px-4 py-8 h-screen flex flex-col">
      <h1 className="text-2xl md:text-3xl font-bold mb-4">Explorador de Modelos</h1>

      {/* Categorías */}
      <div className="mb-4 flex gap-4 border-b overflow-x-auto">
        {Object.keys(categorias).map((cat) => (
          <button
            key={cat}
            onClick={() => {
              setCategoria(cat)
              setPipeline(null)
            }}
            className={`pb-2 px-1 text-sm md:text-base whitespace-nowrap ${
              categoria === cat ? "border-b-2 border-[#FB923C] text-[#FB923C]" : "text-gray-500 hover:text-[#FB923C]"
            }`}
          >
            {cat.toUpperCase()}
          </button>
        ))}
      </div>

      {/* Pipelines */}
      <div className="mb-4 flex gap-2 overflow-x-auto">
        {(categorias[categoria] || []).map((p) => (
          <button
            key={p}
            onClick={() => setPipeline(p)}
            className={`px-3 py-1 text-xs rounded-full border ${
              pipeline === p ? "bg-[#FB923C] text-white" : "text-gray-600 border-gray-300 hover:bg-orange-100"
            }`}
          >
            {p}
          </button>
        ))}
        {pipeline && (
          <button onClick={() => setPipeline(null)} className="ml-2 text-xs text-gray-400 underline">
            Limpiar filtro
          </button>
        )}
      </div>

      {/* Mensaje de advertencia para categorías que no sean nlp */}
      {categoria !== "nlp" && (
        <div className="mb-4 p-3 bg-red-50 border border-red-200 rounded-lg">
          <p className="text-red-600 text-sm font-medium">
            ⚠️ Los notebooks de esta categoría no funcionan correctamente. Se recomienda usar la categoría NLP.
          </p>
        </div>
      )}

      {/* Buscador */}
      <div className="relative mb-4">
        <Search className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-400" />
        <Input
          placeholder="Buscar modelos"
          className="pl-10"
          value={searchTerm}
          onChange={(e) => setSearchTerm(e.target.value)}
        />
      </div>

      {/* Resultados con scroll */}
      <div ref={scrollRef} className="flex-1 overflow-y-auto pr-1 space-y-6">
        {error && <p className="text-center text-red-500 py-12">{error}</p>}

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
          {models.map((model) => (
            <Link key={model.id} href={`/modelo/${encodeURIComponent(model.id)}`} className="animate-fadeIn">
              <Card className="h-full hover:shadow-md transition-shadow border-gray-200 hover:border-[#FB923C]">
                <CardContent className="p-4">
                  <div className="w-12 h-12 bg-orange-100 rounded-full flex items-center justify-center mb-4">
                    <Search className="h-6 w-6 text-[#F97316]" />
                  </div>
                  <h3 className="font-semibold text-lg mb-2">{model.name}</h3>
                  <p className="text-gray-600 text-sm mb-3">{model.description}</p>
                  <div className="flex flex-wrap gap-2">
                    {model.tags.map((tag) => (
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

        {loading && (
          <div className="flex justify-center mt-4">
            <Loader2 className="h-5 w-5 text-[#F97316] animate-spin" />
          </div>
        )}

        {hasMore && <div ref={sentinelRef} className="h-10 mt-4" />}
        {!hasMore && !loading && models.length > 0 && (
          <p className="text-center text-gray-400 mt-6">No hay más modelos.</p>
        )}

        {!loading && models.length === 0 && (
          <div className="text-center py-12">
            <p className="text-gray-500">No se encontraron modelos con los filtros seleccionados.</p>
          </div>
        )}
      </div>
    </div>
  )
}