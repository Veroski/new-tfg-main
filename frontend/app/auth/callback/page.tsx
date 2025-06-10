"use client"

import { useEffect, useState } from "react"
import { useRouter } from "next/navigation"
import { useAuth } from "@/contexts/auth-context"

export default function AuthCallbackPage() {
  const router = useRouter()
  const { setTokenFromOAuth, loading, user } = useAuth()
  const [tokenSet, setTokenSet] = useState(false)

  useEffect(() => {
    const token = new URLSearchParams(window.location.search).get("token")
    if (token) {
      setTokenFromOAuth(token)
      setTokenSet(true) // para indicar que ya se hizo el intento
    } else {
      router.push("/auth/login")
    }
  }, [])

  useEffect(() => {
    if (tokenSet && !loading) {
      if (user) {
        router.push("/perfil")
      } else {
        router.push("/auth/login")
      }
    }
  }, [loading, user, tokenSet])

  return <p className="text-center mt-20">Iniciando sesión con Google...</p>
}