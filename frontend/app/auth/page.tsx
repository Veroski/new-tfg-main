"use client"
import { useState } from "react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { useToast } from "@/hooks/use-toast"
import { iniciarLogin } from "@/lib/api"
import { CheckCircle, Zap, Shield, Users, ArrowRight } from "lucide-react"

export default function AuthPage() {
  const [isLoading, setIsLoading] = useState(false)
  const { toast } = useToast()

  const handleGoogleLogin = async () => {
    setIsLoading(true)
    try {
      const loginUrl = iniciarLogin()
      window.location.href = loginUrl
    } catch (error) {
      toast({
        title: "Error",
        description: "Ha ocurrido un error. Inténtalo de nuevo.",
        variant: "destructive",
      })
      setIsLoading(false)
    }
  }

  const features = [
    {
      icon: <Zap className="h-5 w-5 text-[#F97316]" />,
      title: "Automatización Inteligente",
      description: "Optimiza tus procesos con IA avanzada",
    },
    {
      icon: <Shield className="h-5 w-5 text-[#F97316]" />,
      title: "Seguridad Garantizada",
      description: "Protección de datos de nivel empresarial",
    },
    {
      icon: <Users className="h-5 w-5 text-[#F97316]" />,
      title: "Colaboración Fluida",
      description: "Trabaja en equipo sin complicaciones",
    },
  ]

  return (
    <div className="min-h-[calc(100vh-5rem)] bg-gradient-to-br from-orange-50 to-white">
      <div className="container mx-auto px-4 py-8">
        <div className="grid lg:grid-cols-2 gap-12 items-center">
          {/* Left Column - Content */}
          <div className="space-y-8">
            <div className="space-y-4">
              <Badge variant="outline" className="text-[#F97316] border-[#F97316]">
                Plataforma de Automatización
              </Badge>
              <h1 className="text-4xl lg:text-5xl font-bold text-gray-900 leading-tight">
                Transforma tu flujo de trabajo con <span className="text-[#F97316]">ColabAutomation</span>
              </h1>
              
            </div>

            
          </div>

          {/* Right Column - Auth Card */}
          <div className="flex justify-center">
            <Card className="w-full max-w-md shadow-xl border-0 bg-white/80 backdrop-blur-sm">
              <CardHeader className="text-center space-y-4 pb-6">
                <div className="mx-auto w-16 h-16 bg-[#F97316] rounded-full flex items-center justify-center">
                  <CheckCircle className="h-8 w-8 text-white" />
                </div>
                <div>
                  <CardTitle className="text-2xl font-bold text-gray-900">¡Bienvenido!</CardTitle>
                  <CardDescription className="text-gray-600 mt-2">
                    Accede con tu cuenta de Google para comenzar
                  </CardDescription>
                </div>
              </CardHeader>

              <CardContent className="space-y-6">
                <Button
                  onClick={handleGoogleLogin}
                  disabled={isLoading}
                  className="w-full h-12 bg-white hover:bg-gray-50 text-gray-700 border border-gray-300 shadow-sm transition-all duration-200 hover:shadow-md"
                >
                  <div className="flex items-center justify-center space-x-3">
                    <svg className="w-5 h-5" viewBox="0 0 24 24">
                      <path
                        fill="#4285F4"
                        d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z"
                      />
                      <path
                        fill="#34A853"
                        d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z"
                      />
                      <path
                        fill="#FBBC05"
                        d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z"
                      />
                      <path
                        fill="#EA4335"
                        d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z"
                      />
                    </svg>
                    <span className="font-medium">{isLoading ? "Conectando..." : "Continuar con Google"}</span>
                  </div>
                </Button>

                <div className="text-center space-y-3">
                  <p className="text-sm text-gray-500">
                    Al continuar, aceptas nuestros términos de servicio y política de privacidad
                  </p>

                  <div className="flex items-center justify-center space-x-2 text-sm text-gray-600">
                    <span>¿Nuevo en ColabAutomation?</span>
                    <ArrowRight className="h-4 w-4" />
                    <span className="text-[#F97316] font-medium">Se creará tu cuenta automáticamente</span>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>
        </div>
      </div>
    </div>
  )
}