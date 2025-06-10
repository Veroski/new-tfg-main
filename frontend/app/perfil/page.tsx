/* eslint-disable react-hooks/exhaustive-deps */
"use client"

import { useState, useEffect } from "react"
import { useRouter } from "next/navigation"

import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Separator } from "@/components/ui/separator"

import { useAuth } from "@/contexts/auth-context"
import { useToast } from "@/hooks/use-toast"

import { User, Mail, Settings, LogOut, Edit, Eye, EyeOff, Sparkles } from "lucide-react"

import { getHfToken, putHfToken, postHfToken } from "@/lib/api"

/**
 * 📄 PerfilPage
 * -------------
 */
export default function PerfilPage() {
  const { user, logout } = useAuth()
  const router = useRouter()
  const { toast } = useToast()

  // --------------------------
  // Estado de la información personal
  // --------------------------
  const [isEditing, setIsEditing] = useState(false)
  const [formData, setFormData] = useState({
    name: user?.name || "",
    email: user?.email || "",
  })

  // --------------------------
  // Estado del token de Hugging Face
  // --------------------------
  const [hfToken, setHfToken] = useState("")
  const [isEditingToken, setIsEditingToken] = useState(false)
  const [showToken, setShowToken] = useState(false)
  const [tokenInput, setTokenInput] = useState("")
  const [hasExistingToken, setHasExistingToken] = useState(false)
  const [isLoadingToken, setIsLoadingToken] = useState(true)

  // ---------------------------------------------------------------------------
  // 1️⃣  Cargar token al montar el componente (y cuando `user` cambie).
  // ---------------------------------------------------------------------------
  useEffect(() => {
    if (!user) {
      router.push("/auth")
    } else {
      loadHfToken()
    }
  }, [user])

  /**
   * Obtiene el token del backend. Si el backend envía `null` (texto o real), se
   * interpreta como *sin token*.
   */
  const loadHfToken = async () => {
    try {
      setIsLoadingToken(true)
      const token = await getHfToken()

      // 🚫 Evitamos el bug de "null" como string
      if (token && token !== "null") {
        // 🔧 Limpiamos las comillas que puedan venir del backend
        const cleanToken = token.replace(/^["']|["']$/g, "")
        setHfToken(cleanToken)
        setHasExistingToken(true)
      } else {
        setHfToken("")
        setHasExistingToken(false)
      }
    } catch (error) {
      console.error("Error loading HF token:", error)
    } finally {
      setIsLoadingToken(false)
    }
  }

  // ---------------------------------------------------------------------------
  // 2️⃣  Manejo de información personal
  // ---------------------------------------------------------------------------
  const handleSave = () => {
    toast({
      title: "Perfil actualizado",
      description: "Tus cambios han sido guardados exitosamente.",
    })
    setIsEditing(false)
  }

  // ---------------------------------------------------------------------------
  // 3️⃣  Guardar / actualizar token de HF
  // ---------------------------------------------------------------------------
  const handleSaveToken = async () => {
    try {
      if (hasExistingToken) {
        await putHfToken(tokenInput.trim())
        toast({
          title: "Token actualizado",
          description: "Tu token de Hugging Face ha sido actualizado exitosamente.",
        })
      } else {
        await postHfToken(tokenInput.trim())
        toast({
          title: "Token guardado",
          description: "Tu token de Hugging Face ha sido guardado exitosamente.",
        })
        setHasExistingToken(true)
      }

      setHfToken(tokenInput.trim())
      setIsEditingToken(false)
      setTokenInput("")
    } catch (error) {
      toast({
        title: "Error",
        description: "No se pudo guardar el token. Inténtalo de nuevo.",
        variant: "destructive",
      })
    }
  }

  const handleCancelToken = () => {
    setTokenInput("")
    setIsEditingToken(false)
  }

  const handleEditToken = () => {
    setTokenInput(hfToken)
    setIsEditingToken(true)
  }

  /**
   * 🔒 Devuelve una cadena con la misma longitud del token original compuesta
   * totalmente de asteriscos. Soluciona el problema de que se mostraban los
   * últimos 4 caracteres.
   */
  const maskToken = (token: string) => "*".repeat(token.length || 8)

  const handleLogout = () => {
    logout()
    router.push("/")
    toast({
      title: "Sesión cerrada",
      description: "Has cerrado sesión exitosamente.",
    })
  }

  // ---------------------------------------------------------------------------
  // 4️⃣  Renderizado
  // ---------------------------------------------------------------------------
  if (!user) {
    return (
      <div className="flex items-center justify-center h-screen">
        <p className="text-gray-500 text-lg">Redirigiendo…</p>
      </div>
    )
  }

  return (
    <div className="container mx-auto px-4 py-8 max-w-2xl">
      <div className="space-y-6">
        {/* Encabezado */}
        <header>
          <h1 className="text-2xl md:text-3xl font-bold">Mi Perfil</h1>
          <p className="text-gray-600">Gestiona tu información personal y configuración de cuenta</p>
        </header>

        {/* ---------------- Información Personal ---------------- */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <User className="h-5 w-5" /> Información Personal
            </CardTitle>
            <CardDescription>Actualiza tu información personal y datos de contacto</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            {/* Nombre */}
            <div className="space-y-2">
              <Label htmlFor="name">Nombre</Label>
              {isEditing ? (
                <Input
                  id="name"
                  value={formData.name}
                  onChange={(e) => setFormData({ ...formData, name: e.target.value })}
                />
              ) : (
                <div className="p-2 bg-gray-50 rounded-md">{user.name}</div>
              )}
            </div>

            {/* Email */}
            <div className="space-y-2">
              <Label htmlFor="email">Email</Label>
              {isEditing ? (
                <Input
                  id="email"
                  type="email"
                  value={formData.email}
                  onChange={(e) => setFormData({ ...formData, email: e.target.value })}
                />
              ) : (
                <div className="p-2 bg-gray-50 rounded-md flex items-center gap-2">
                  <Mail className="h-4 w-4 text-gray-500" />
                  {user.email}
                </div>
              )}
            </div>

            {/* Botones */}
            <div className="flex gap-2 pt-4">
              {isEditing ? (
                <>
                  <Button onClick={handleSave} className="bg-[#F97316] hover:bg-[#EA580C]">
                    Guardar Cambios
                  </Button>
                  <Button variant="outline" onClick={() => setIsEditing(false)}>
                    Cancelar
                  </Button>
                </>
              ) : (
                <Button onClick={() => setIsEditing(true)} variant="outline">
                  <Settings className="h-4 w-4 mr-2" /> Editar Perfil
                </Button>
              )}
            </div>
          </CardContent>
        </Card>

        {/* ---------------- Token de Hugging Face ---------------- */}
        <Card className={!hasExistingToken && !isLoadingToken ? "ring-2 ring-orange-200 animate-pulse" : ""}>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Sparkles className="h-5 w-5 text-orange-500" /> Token de Hugging Face
              {!hasExistingToken && !isLoadingToken && (
                <span className="text-xs bg-orange-100 text-orange-600 px-2 py-1 rounded-full animate-bounce">
                  ¡Recomendado!
                </span>
              )}
            </CardTitle>
            <CardDescription>Configura tu token de Hugging Face para acceder a los modelos de IA</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            {/* Loading */}
            {isLoadingToken ? (
              <div className="p-4 text-center text-gray-500">Cargando token...</div>
            ) : (
              <>
                <div className="space-y-2">
                  <Label htmlFor="hf-token">Token</Label>
                  {isEditingToken ? (
                    <div className="space-y-3">
                      <Input
                        id="hf-token"
                        type="password"
                        placeholder="Ingresa tu token de Hugging Face"
                        value={tokenInput}
                        onChange={(e) => setTokenInput(e.target.value)}
                        className={!hasExistingToken ? "ring-2 ring-orange-300 animate-pulse" : ""}
                      />
                      <div className="flex gap-2">
                        <Button
                          onClick={handleSaveToken}
                          className="bg-[#F97316] hover:bg-[#EA580C]"
                          disabled={!tokenInput.trim()}
                        >
                          {hasExistingToken ? "Actualizar" : "Guardar"}
                        </Button>
                        <Button variant="outline" onClick={handleCancelToken}>
                          Cancelar
                        </Button>
                      </div>
                    </div>
                  ) : (
                    <div className="flex items-center gap-2">
                      {hfToken ? (
                        <>
                          <div className="flex-1 p-2 bg-gray-50 rounded-md font-mono text-sm select-all">
                            {showToken ? hfToken : maskToken(hfToken)}
                          </div>
                          <Button variant="ghost" size="sm" onClick={() => setShowToken(!showToken)}>
                            {showToken ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
                          </Button>
                          <Button variant="ghost" size="sm" onClick={handleEditToken}>
                            <Edit className="h-4 w-4" />
                          </Button>
                        </>
                      ) : (
                        <div className="flex-1 flex items-center justify-between p-3 border-2 border-dashed border-orange-300 rounded-md bg-orange-50">
                          <span className="text-orange-600 font-medium animate-pulse">
                            ¡Se recomienda configurar tu token de Hugging Face!
                          </span>
                          <Button
                            onClick={() => setIsEditingToken(true)}
                            className="bg-[#F97316] hover:bg-[#EA580C] animate-bounce"
                          >
                            <Sparkles className="h-4 w-4 mr-2" /> Añadir Token
                          </Button>
                        </div>
                      )}
                    </div>
                  )}
                </div>

                {/* Ayuda para obtener el token */}
                {!hfToken && !isEditingToken && (
                  <div className="p-3 bg-blue-50 border border-blue-200 rounded-md">
                    <p className="text-sm text-blue-700">
                      <strong>¿Cómo obtener tu token?</strong>
                      <br /> 1. Ve a{" "}
                      <a
                        href="https://huggingface.co/settings/tokens"
                        target="_blank"
                        rel="noopener noreferrer"
                        className="underline hover:text-blue-800"
                      >
                        huggingface.co/settings/tokens
                      </a>
                      <br /> 2. Crea un nuevo token con permisos de lectura
                      <br /> 3. Copia y pega el token aquí
                    </p>
                  </div>
                )}
              </>
            )}
          </CardContent>
        </Card>

        {/* ---------------- Cerrar sesión ---------------- */}
        <Card>
          <CardContent>
            <Separator className="mb-4" />
            <div className="flex items-center justify-between">
              <div>
                <h3 className="font-medium">Cerrar Sesión</h3>
                <p className="text-sm text-gray-600">Cierra tu sesión actual</p>
              </div>
              <Button variant="destructive" onClick={handleLogout}>
                <LogOut className="h-4 w-4 mr-2" /> Cerrar Sesión
              </Button>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  )
}
