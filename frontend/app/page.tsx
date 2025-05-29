import Link from "next/link"
import { Search, BookOpen } from "lucide-react"

export default function Home() {
  return (
    <div className="container mx-auto px-4 py-12">
      <section className="max-w-4xl mx-auto">
        <h1 className="text-3xl md:text-4xl font-bold text-center mb-12">
          Bienvenido a <span className="text-[#F97316]">ColabAutomation</span>
        </h1>

        <div className="flex justify-center max-w-2xl mx-auto">
          {/* Explorar Modelos */}
          <Link
            href="/explorar"
            className="bg-white hover:bg-orange-50 border border-gray-200 rounded-xl p-6 shadow-sm hover:shadow-md transition-all group"
          >
            <div className="flex flex-col items-center text-center gap-4">
              <div className="w-16 h-16 bg-orange-100 rounded-full flex items-center justify-center group-hover:bg-orange-200 transition-colors">
                <Search className="h-8 w-8 text-[#F97316]" />
              </div>
              <h2 className="text-xl font-semibold">Explorar Modelos</h2>
              <p className="text-gray-600">Busca y filtra entre nuestra colección de modelos disponibles</p>
            </div>
          </Link>
        </div>

      </section>
    </div>
  )
}
