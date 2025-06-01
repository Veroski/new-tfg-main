"use client"

import React, { createContext, useContext, useState, useEffect } from "react"
import { getUsuarioActual, API_BASE_URL } from "@/lib/api"


interface User {
  id: string
  email: string
  name: string
  avatar?: string
}

interface AuthContextType {
  user: User | null
  token: string | null
  loading: boolean
  login: (email: string, password: string) => Promise<boolean>
  register: (email: string, password: string, name: string) => Promise<boolean>
  logout: () => void
  setTokenFromOAuth: (token: string) => void
}

const AuthContext = createContext<AuthContextType | undefined>(undefined)

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const [user, setUser] = useState<User | null>(null)
  const [token, setToken] = useState<string | null>(null)
  const [loading, setLoading] = useState(true)

  // Cargar token y usuario desde localStorage
  useEffect(() => {
    const storedToken = localStorage.getItem("auth_token")
    if (storedToken) {
      setToken(storedToken)
      fetchUser(storedToken)
    } else {
      setLoading(false)
    }
  }, [])

  // 🧠 Obtener el usuario actual desde el backend
  const fetchUser = async (jwtToken: string) => {
    try {
      const res = await fetch(getUsuarioActual(), {
        headers: {
          Authorization: `Bearer ${jwtToken}`,
        },
      })

      if (res.ok) {
        const data = await res.json()
        setUser(data)
      } else {
        logout()
      }
    } catch (err) {
      logout()
    } finally {
      setLoading(false)
    }
  }


  // 🔐 Login con usuario/contraseña
  const login = async (email: string, password: string): Promise<boolean> => {
    try {
      const res = await fetch(`${API_BASE_URL}/auth/login`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ email, password }),
      })

      if (!res.ok) return false

      const data = await res.json()
      const jwtToken = data.token
      localStorage.setItem("auth_token", jwtToken)
      setToken(jwtToken)
      await fetchUser(jwtToken)

      return true
    } catch (err) {
      return false
    }
  }

  // 🆕 Registro de usuario
  const register = async (email: string, password: string, name: string): Promise<boolean> => {
    try {
      const res = await fetch(`${API_BASE_URL}/auth/register`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ email, password, name }),
      })

      if (!res.ok) return false

      const data = await res.json()
      const jwtToken = data.token
      localStorage.setItem("auth_token", jwtToken)
      setToken(jwtToken)
      await fetchUser(jwtToken)

      return true
    } catch (err) {
      return false
    }
  }

  // 🔓 Logout
  const logout = () => {
    localStorage.removeItem("auth_token")
    setUser(null)
    setToken(null)
  }

  // 🔁 Login desde Google OAuth (token ya generado)
  const setTokenFromOAuth = (newToken: string) => {
    localStorage.setItem("auth_token", newToken)
    setToken(newToken)
    fetchUser(newToken)
  }

  return (
    <AuthContext.Provider value={{ user, token, loading, login, register, logout, setTokenFromOAuth }}>
      {children}
    </AuthContext.Provider>
  )
}

export function useAuth() {
  const context = useContext(AuthContext)
  if (context === undefined) {
    throw new Error("useAuth must be used within an AuthProvider")
  }
  return context
}
