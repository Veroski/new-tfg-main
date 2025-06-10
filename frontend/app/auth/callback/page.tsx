"use client"

import { useEffect } from "react"
import { useRouter } from "next/navigation"
import { useAuth } from "@/contexts/auth-context"

export default function AuthCallbackPage() {
  const router = useRouter()
  const { setTokenFromOAuth } = useAuth()

  useEffect(() => {
    const token = new URLSearchParams(window.location.search).get("token")
    if (token) {
      setTokenFromOAuth(token)
      router.push("/perfil")
    } else {
      router.push("/auth/login")
    }
  }, [])

  return <p className="text-center mt-20">Iniciando sesión con Google...</p>
}