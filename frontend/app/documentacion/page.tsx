import Link from "next/link"
import { BookOpen, FileText, HelpCircle, Search } from "lucide-react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"

export default function Documentacion() {
  return (
    <div className="container mx-auto px-4 py-8">
      <h1 className="text-2xl md:text-3xl font-bold mb-4">Documentación</h1>
      <p className="text-gray-600 mb-8">Consulta guías y recursos para utilizar los modelos de manera efectiva</p>

      <div className="relative mb-8">
        <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400" />
        <Input placeholder="Buscar en la documentación..." className="pl-10" />
      </div>

      <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-3">
        <Card className="hover:shadow-md transition-shadow">
          <CardHeader className="pb-3">
            <CardTitle className="flex items-center gap-2">
              <BookOpen className="h-5 w-5 text-[#F97316]" />
              Guías de Inicio
            </CardTitle>
            <CardDescription>Aprende los conceptos básicos para comenzar a utilizar los modelos</CardDescription>
          </CardHeader>
          <CardContent>
            <ul className="space-y-2">
              <li>
                <Link href="#" className="text-[#F97316] hover:underline">
                  Introducción a ColabAutomation
                </Link>
              </li>
              <li>
                <Link href="#" className="text-[#F97316] hover:underline">
                  Cómo descargar modelos
                </Link>
              </li>
              <li>
                <Link href="#" className="text-[#F97316] hover:underline">
                  Requisitos del sistema
                </Link>
              </li>
              <li>
                <Link href="#" className="text-[#F97316] hover:underline">
                  Primeros pasos con modelos de NLP
                </Link>
              </li>
              <li>
                <Link href="#" className="text-[#F97316] hover:underline">
                  Primeros pasos con modelos de visión
                </Link>
              </li>
            </ul>
          </CardContent>
        </Card>

        <Card className="hover:shadow-md transition-shadow">
          <CardHeader className="pb-3">
            <CardTitle className="flex items-center gap-2">
              <FileText className="h-5 w-5 text-[#F97316]" />
              Tutoriales
            </CardTitle>
            <CardDescription>Guías paso a paso para implementar casos de uso específicos</CardDescription>
          </CardHeader>
          <CardContent>
            <ul className="space-y-2">
              <li>
                <Link href="#" className="text-[#F97316] hover:underline">
                  Generación de texto con GPT-4
                </Link>
              </li>
              <li>
                <Link href="#" className="text-[#F97316] hover:underline">
                  Creación de imágenes con Stable Diffusion
                </Link>
              </li>
              <li>
                <Link href="#" className="text-[#F97316] hover:underline">
                  Reconocimiento de voz con Whisper
                </Link>
              </li>
              <li>
                <Link href="#" className="text-[#F97316] hover:underline">
                  Clasificación de imágenes con ResNet
                </Link>
              </li>
              <li>
                <Link href="#" className="text-[#F97316] hover:underline">
                  Embeddings de texto con BERT
                </Link>
              </li>
            </ul>
          </CardContent>
        </Card>

        <Card className="hover:shadow-md transition-shadow">
          <CardHeader className="pb-3">
            <CardTitle className="flex items-center gap-2">
              <HelpCircle className="h-5 w-5 text-[#F97316]" />
              Preguntas Frecuentes
            </CardTitle>
            <CardDescription>Respuestas a las preguntas más comunes</CardDescription>
          </CardHeader>
          <CardContent>
            <ul className="space-y-2">
              <li>
                <Link href="#" className="text-[#F97316] hover:underline">
                  ¿Cómo puedo optimizar el rendimiento de los modelos?
                </Link>
              </li>
              <li>
                <Link href="#" className="text-[#F97316] hover:underline">
                  ¿Qué hardware se recomienda para cada tipo de modelo?
                </Link>
              </li>
              <li>
                <Link href="#" className="text-[#F97316] hover:underline">
                  ¿Cómo puedo contribuir con mis propios modelos?
                </Link>
              </li>
              <li>
                <Link href="#" className="text-[#F97316] hover:underline">
                  ¿Los modelos son compatibles con mi framework?
                </Link>
              </li>
              <li>
                <Link href="#" className="text-[#F97316] hover:underline">
                  ¿Cómo reportar problemas o errores?
                </Link>
              </li>
            </ul>
          </CardContent>
        </Card>
      </div>

      <div className="mt-12">
        <h2 className="text-xl font-semibold mb-6">Documentación por Categoría</h2>

        <div className="grid gap-6 md:grid-cols-2">
          <Card className="hover:shadow-md transition-shadow">
            <CardHeader className="pb-3">
              <CardTitle>Modelos de Lenguaje Natural (NLP)</CardTitle>
            </CardHeader>
            <CardContent>
              <ul className="space-y-2">
                <li>
                  <Link href="#" className="text-[#F97316] hover:underline">
                    Guía completa de GPT-4
                  </Link>
                </li>
                <li>
                  <Link href="#" className="text-[#F97316] hover:underline">
                    Trabajando con LLaMA 2
                  </Link>
                </li>
                <li>
                  <Link href="#" className="text-[#F97316] hover:underline">
                    Embeddings con BERT
                  </Link>
                </li>
                <li>
                  <Link href="#" className="text-[#F97316] hover:underline">
                    Fine-tuning de modelos de lenguaje
                  </Link>
                </li>
              </ul>
            </CardContent>
          </Card>

          <Card className="hover:shadow-md transition-shadow">
            <CardHeader className="pb-3">
              <CardTitle>Modelos de Visión por Computadora</CardTitle>
            </CardHeader>
            <CardContent>
              <ul className="space-y-2">
                <li>
                  <Link href="#" className="text-[#F97316] hover:underline">
                    Guía de Stable Diffusion
                  </Link>
                </li>
                <li>
                  <Link href="#" className="text-[#F97316] hover:underline">
                    Clasificación con ResNet
                  </Link>
                </li>
                <li>
                  <Link href="#" className="text-[#F97316] hover:underline">
                    Detección de objetos
                  </Link>
                </li>
                <li>
                  <Link href="#" className="text-[#F97316] hover:underline">
                    Segmentación de imágenes
                  </Link>
                </li>
              </ul>
            </CardContent>
          </Card>

          <Card className="hover:shadow-md transition-shadow">
            <CardHeader className="pb-3">
              <CardTitle>Modelos de Audio</CardTitle>
            </CardHeader>
            <CardContent>
              <ul className="space-y-2">
                <li>
                  <Link href="#" className="text-[#F97316] hover:underline">
                    Transcripción con Whisper
                  </Link>
                </li>
                <li>
                  <Link href="#" className="text-[#F97316] hover:underline">
                    Reconocimiento de voz con Wav2Vec
                  </Link>
                </li>
                <li>
                  <Link href="#" className="text-[#F97316] hover:underline">
                    Clasificación de audio
                  </Link>
                </li>
                <li>
                  <Link href="#" className="text-[#F97316] hover:underline">
                    Generación de audio
                  </Link>
                </li>
              </ul>
            </CardContent>
          </Card>

          <Card className="hover:shadow-md transition-shadow">
            <CardHeader className="pb-3">
              <CardTitle>Modelos Multimodales</CardTitle>
            </CardHeader>
            <CardContent>
              <ul className="space-y-2">
                <li>
                  <Link href="#" className="text-[#F97316] hover:underline">
                    Introducción a CLIP
                  </Link>
                </li>
                <li>
                  <Link href="#" className="text-[#F97316] hover:underline">
                    Búsqueda de imágenes por texto
                  </Link>
                </li>
                <li>
                  <Link href="#" className="text-[#F97316] hover:underline">
                    Generación de imágenes con DALL-E
                  </Link>
                </li>
                <li>
                  <Link href="#" className="text-[#F97316] hover:underline">
                    Modelos de visión y lenguaje
                  </Link>
                </li>
              </ul>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  )
}
